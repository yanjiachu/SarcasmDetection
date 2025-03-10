import json
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, classification_report

# 定义超参数
batch_size = 32
learning_rate = 1e-4
dropout_prob = 0.2
patience_num = 3    # 早停阈值
draw_step = 3       # 绘制loss和acc的图像的间隔
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
model_path = '../../bert-base-chinese'
# model_path = '../../chinese-macbert-base'
best_model_path = '../../models/detect/mynet_bert_4.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(dropout_prob)
        # self.diff_layer = torch.nn.Sequential(
        #     torch.nn.Linear(768 * 2, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(dropout_prob),
        #     torch.nn.Linear(256, num_labels)
        # )
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv1d(768, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(256, num_labels)
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv1d(768, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(256, num_labels)
        )
        self.classifier = torch.nn.Linear(256 * 2, num_labels)
        # 冻结参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, comment_input_ids, comment_attention_mask, context_input_ids, context_attention_mask, labels=None):
        comment_outputs = self.bert(
            input_ids=comment_input_ids,
            attention_mask=comment_attention_mask
        )
        comment_cls = comment_outputs.last_hidden_state[:, 0, :]
        context_outputs = self.bert(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_cls = context_outputs.last_hidden_state[:, 0, :]

        # combined = torch.cat([comment_cls, context_cls], dim=1)
        # logits = self.diff_layer(combined)

        comment_cnn_out = self.cnn1(comment_cls.unsqueeze(2)).squeeze(2)
        context_cnn_out = self.cnn2(context_cls.unsqueeze(2)).squeeze(2)
        combined = torch.cat([comment_cnn_out, context_cnn_out], dim=1)

        logits = self.classifier(combined)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return logits, loss
        else:
            return logits

# 定义数据集类
class SarcasmDataset(Dataset):
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

        # 获取话题内容
        topic_content = self.topic_dict.get(topic_id, {})
        topic_title = topic_content.get('topicTitle', '')
        topic_text_content = topic_content.get('topicContent', '')

        # 构造双通道输入
        comment_input = review
        context_input = f"{topic_title} [SEP] {topic_text_content}"

        # 编码
        comment_encoding = self.tokenizer(
            comment_input,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )
        context_encoding = self.tokenizer(
            context_input,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'comment_input_ids': comment_encoding['input_ids'].squeeze(),
            'comment_attention_mask': comment_encoding['attention_mask'].squeeze(),
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(is_sarcasm, dtype=torch.long)
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

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, roc_auc, path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.savefig(path)
    plt.show()
    plt.close()

# 主函数
if __name__ == '__main__':
    # 加载数据
    train_data = load_data_dev(train_path)
    topic_data = load_topic_data(train_topic_path)

    # 分割数据集为训练集和测试集
    train_data, test_data = train_test_split(train_data, test_size=test_size, random_state=42)

    # 初始化 BERT 分词器
    tokenizer = BertTokenizerFast.from_pretrained(model_path, use_fast=True)

    # 创建数据集
    train_dataset = SarcasmDataset(train_data, topic_data, tokenizer)
    test_dataset = SarcasmDataset(test_data, topic_data, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = MyModel(num_labels=2, dropout_prob=dropout_prob)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 初始化列表
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # 早停机制
    patience = patience_num
    best_accuracy = 0.0

    # 训练循环
    print("Training...")
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 训练每一批次
        for batch in train_loader:
            comment_input_ids = batch['comment_input_ids'].to(device)
            comment_attention_mask = batch['comment_attention_mask'].to(device)
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 获取 logits 和 loss
            logits, loss = model(comment_input_ids, comment_attention_mask, context_input_ids, context_attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 计算并保存训练精度
        train_accuracy = total_correct / total_samples
        train_accuracies.append(train_accuracy)

        # 测试阶段
        model.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                comment_input_ids = batch['comment_input_ids'].to(device)
                comment_attention_mask = batch['comment_attention_mask'].to(device)
                context_input_ids = batch['context_input_ids'].to(device)
                context_attention_mask = batch['context_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 获取 logits 和 loss
                logits, test_loss = model(comment_input_ids, comment_attention_mask, context_input_ids, context_attention_mask, labels=labels)
                predictions = torch.argmax(logits, dim=1)

                # 累加测试损失
                total_test_loss += test_loss.item()

                # 计算准确率
                total_test_correct += (predictions == labels).sum().item()
                total_test_samples += labels.size(0)

        # 计算并保存测试精度
        test_accuracy = total_test_correct / total_test_samples
        test_accuracies.append(test_accuracy)

        # 计算并保存测试损失
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # 打印结果
        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy * 100:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Acc: {test_accuracy * 100:.2f}%")

        # 早停机制
        if test_accuracy > best_accuracy:
            patience = patience_num
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")


    # 加载最佳模型并评估
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    true_labels = []
    pred_probs = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            comment_input_ids = batch['comment_input_ids'].to(device)
            comment_attention_mask = batch['comment_attention_mask'].to(device)
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取模型输出
            logits = model(comment_input_ids, comment_attention_mask, context_input_ids, context_attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            pred_probs.extend(probabilities[:, 1].cpu().numpy())  # 正类概率（讽刺类别）
            pred_labels.extend(preds.cpu().numpy())  # 预测标签
            true_labels.extend(labels.cpu().numpy())

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, roc_auc, path=f'../../ROC/mynet_4.png')

    # 计算召回率、F1分数等指标
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    classification_rep = classification_report(true_labels, pred_labels, target_names=['Non-Sarcastic', 'Sarcastic'])

    # 输出结果
    print(f"AUC: {roc_auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")