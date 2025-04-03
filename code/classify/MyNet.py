import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

# 定义超参数
batch_size = 32
learning_rate = 2e-5
dropout_prob = 0.25
patience_num = 3    # 早停阈值
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
bert_path = '../../bert-base-chinese'
# bert_path = '../../chinese-macbert-base'
best_model_path = '../../models/classify/SC_Hybrid.pth'
pic_path = '../../ConfusionMatrix/SC_Hybrid.png'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

class SC_Hybrid(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(SC_Hybrid, self).__init__()

        # CNN 分支
        self.conv3 = torch.nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.conv7 = torch.nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3)

        # CNN 特征降维层
        self.cnn_fc = torch.nn.Linear(hidden_size * 3, hidden_size)  # 将 CNN 特征从 768 降维到 256

        # Bi-LSTM 分支
        self.lstm = torch.nn.LSTM(input_size, hidden_size // 2, num_layers=2, bidirectional=True, batch_first=True)

        # 动态融合门
        self.fusion_gate_cnn = torch.nn.Linear(hidden_size, 1)  # 为 CNN 特征计算权重
        self.fusion_gate_lstm = torch.nn.Linear(hidden_size, 1)  # 为 LSTM 特征计算权重

        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        # CNN 分支
        x_cnn = x.transpose(1, 2)  # 调整维度以适应 Conv1d [batch, input_size, seq_len]
        cnn3 = torch.relu(self.conv3(x_cnn)).transpose(1, 2)  # [batch, seq_len, hidden_size]
        cnn5 = torch.relu(self.conv5(x_cnn)).transpose(1, 2)  # [batch, seq_len, hidden_size]
        cnn7 = torch.relu(self.conv7(x_cnn)).transpose(1, 2)  # [batch, seq_len, hidden_size]
        cnn_feat = torch.cat([cnn3, cnn5, cnn7], dim=2)  # [batch, seq_len, hidden_size * 3]

        # CNN 特征降维
        cnn_feat = cnn_feat.mean(dim=1)  # [batch, hidden_size * 3]
        cnn_feat = self.cnn_fc(cnn_feat)  # [batch, hidden_size]

        # Bi-LSTM 分支
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        lstm_feat = lstm_out.mean(dim=1)  # [batch, hidden_size]

        # 动态融合
        gate_cnn = torch.sigmoid(self.fusion_gate_cnn(cnn_feat))  # [batch, 1]
        gate_lstm = torch.sigmoid(self.fusion_gate_lstm(lstm_feat))  # [batch, 1]

        # 加权融合
        final_feat = gate_cnn * cnn_feat + gate_lstm * lstm_feat  # [batch, hidden_size]
        out = self.dropout(final_feat)
        return out

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.model = SC_Hybrid(
            self.bert.config.hidden_size,
            hidden_size,
            dropout_prob
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_size, 256)
        self.classifier = torch.nn.Linear(256, num_labels)

        # 冻结 BERT 参数
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        out = self.model(hidden_states)

        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.classifier(out)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return logits, loss
        else:
            return logits

# 定义数据集类
class SarcasmClassificationDataset(Dataset):
    def __init__(self, data, topic_dict, tokenizer):
        self.data = data
        self.topic_dict = topic_dict
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        topic_id = item['topicId']
        review = item['review']
        is_sarcasm = item['isSarcasm']
        sarcasm_type = item['sarcasmType']

        # 只处理标签为1~6的讽刺文本
        if is_sarcasm == 1 and sarcasm_type is not None:
            label = sarcasm_type - 1  # 转换为从0开始的索引
        else:
            return None  # 跳过非讽刺或标签为None的样本

        # 获取话题内容
        topic_content = self.topic_dict.get(topic_id, {})
        topic_title = topic_content.get('topicTitle', '')
        topic_text_content = topic_content.get('topicContent', '')

        # 拼接评论和话题内容
        input_text = f"{review}[SEP]{topic_text_content}"

        # 使用BERT tokenizer编码
        encoding = self.tokenizer(
            input_text,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 加载评论
def load_data_dev(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 加载话题
def load_topic_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    topic_dict = {item['topicId']: item for item in data}
    return topic_dict

def plot_loss_acc(train_losses, test_losses, train_accuracies, test_accuracies, epoch, path):
    epochs = range(1, epoch + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, test_losses, 'r', label='Test Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
    plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
    plt.title('Test Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)

# 主函数
if __name__ == '__main__':
    # 加载数据
    train_data = load_data_dev(train_path)
    topic_data = load_topic_data(train_topic_path)

    # 筛选出标签有效的数据
    filtered_data = [item for item in train_data if item['isSarcasm'] == 1 and item['sarcasmType'] is not None]

    # 分割数据集为训练集和测试集
    train_data, test_data = train_test_split(filtered_data, test_size=test_size, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)

    # 创建数据集
    train_dataset = SarcasmClassificationDataset(train_data, topic_data, tokenizer)
    test_dataset = SarcasmClassificationDataset(test_data, topic_data, tokenizer)

    # 过滤掉无效的样本
    train_dataset.data = [item for item in train_dataset.data if item is not None]
    test_dataset.data = [item for item in test_dataset.data if item is not None]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    # 定义模型
    model = MyModel(hidden_size=256 ,num_labels=6, dropout_prob=dropout_prob)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 初始化列表
    train_losses = []
    test_losses = []
    train_accuracies = []  # 训练集评论级别准确率
    test_accuracies = []   # 测试集评论级别准确率

    # 早停机制
    patience = patience_num
    best_accuracy = 0.0

    # 训练循环
    # print("Training...")
    # start_time = time.time()
    # for epoch in range(1, num_epochs + 1):
    #     model.train()  # 设置模型为训练模式
    #     total_loss = 0.0
    #     total_correct = 0  # 统计训练集正确预测数
    #     total_samples = 0  # 统计训练集总样本数
    #
    #     # 训练每一批次
    #     for batch in train_loader:
    #         # 合并批次数据
    #         input_ids = []
    #         attention_masks = []
    #         labels = []
    #         for sample in batch:
    #             if sample is not None:
    #                 input_ids.append(sample['input_ids'])
    #                 attention_masks.append(sample['attention_mask'])
    #                 labels.append(sample['label'])
    #
    #         if not input_ids:
    #             continue
    #
    #         input_ids = torch.stack(input_ids).to(device)
    #         attention_masks = torch.stack(attention_masks).to(device)
    #         labels = torch.stack(labels).to(device)
    #
    #         optimizer.zero_grad()
    #
    #         # 获取 logits 和 loss
    #         logits, loss = model(input_ids, attention_masks, labels=labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #         # 计算训练集评论级别准确率
    #         predictions = torch.argmax(logits, dim=1)
    #         total_correct += (predictions == labels).sum().item()
    #         total_samples += labels.size(0)
    #
    #     # 计算并保存训练集损失和准确率
    #     avg_train_loss = total_loss / len(train_loader)
    #     train_losses.append(avg_train_loss)
    #     train_accuracy = total_correct / total_samples
    #     train_accuracies.append(train_accuracy)
    #
    #     # 测试阶段
    #     model.eval()
    #     true_labels = []
    #     pred_labels = []
    #     total_test_loss = 0.0  # 统计测试集损失
    #
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             input_ids = []
    #             attention_masks = []
    #             labels = []
    #             for sample in batch:
    #                 if sample is not None:
    #                     input_ids.append(sample['input_ids'])
    #                     attention_masks.append(sample['attention_mask'])
    #                     labels.append(sample['label'])
    #
    #             if not input_ids:
    #                 continue
    #
    #             input_ids = torch.stack(input_ids).to(device)
    #             attention_masks = torch.stack(attention_masks).to(device)
    #             labels = torch.stack(labels).to(device)
    #
    #             # 获取 logits 和 loss
    #             logits, test_loss = model(input_ids, attention_masks, labels=labels)
    #             predictions = torch.argmax(logits, dim=1)
    #
    #             # 累加测试集损失
    #             total_test_loss += test_loss.item()
    #
    #             # 保存真实标签和预测标签
    #             true_labels.extend(labels.cpu().numpy())
    #             pred_labels.extend(predictions.cpu().numpy())
    #
    #     # 计算测试集损失和准确率
    #     avg_test_loss = total_test_loss / len(test_loader)
    #     test_losses.append(avg_test_loss)
    #     test_accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    #     test_accuracies.append(test_accuracy)
    #
    #     # 打印结果
    #     print(f"Epoch {epoch}/{num_epochs}, "
    #           f"Train Loss: {avg_train_loss:.4f}, "
    #           f"Train Acc: {train_accuracy * 100:.2f}%, "
    #           f"Test Loss: {avg_test_loss:.4f}, "
    #           f"Test Acc: {test_accuracy * 100:.2f}%")
    #
    #     # 早停机制
    #     if test_accuracy > best_accuracy:
    #         patience = patience_num
    #         best_accuracy = test_accuracy
    #         torch.save(model.state_dict(), best_model_path)
    #     else:
    #         patience -= 1
    #         if patience == 0:
    #             print("Early stopping!")
    #             break
    #
    # end_time = time.time()
    # total_training_time = end_time - start_time
    # print(f"Total training time: {total_training_time:.2f} seconds")

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 在测试阶段收集真实标签和预测标签
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = []
            attention_masks = []
            labels = []
            for sample in batch:
                input_ids.append(sample['input_ids'])
                attention_masks.append(sample['attention_mask'])
                labels.append(sample['label'])

            input_ids = torch.stack(input_ids).to(device)
            attention_masks = torch.stack(attention_masks).to(device)
            labels = torch.stack(labels).to(device)

            logits = model(input_ids, attention_masks)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # 计算F1值
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=range(6))

    # 输出混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3, 4, 5, 6],
                yticklabels=[1, 2, 3, 4, 5, 6])
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    plt.title('Confusion Matrix of Sarcasm Types')
    plt.savefig(pic_path)
    plt.close()

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 输出归一化后的混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[1, 2, 3, 4, 5, 6],
                yticklabels=[1, 2, 3, 4, 5, 6])
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    plt.title('Normalized Confusion Matrix of Sarcasm Types')
    plt.savefig(pic_path.replace('.png', '_normalized.png'))
    plt.close()