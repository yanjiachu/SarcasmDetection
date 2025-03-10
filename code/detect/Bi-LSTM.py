import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


# 定义超参数
batch_size = 16
learning_rate = 5e-5
dropout_prob = 0.05
patience_num = 5    # 早停阈值
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
model_path = '../../bert-base-chinese'
best_model_path = '../../models/detect/bi-lstm.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


# 定义Bi-LSTM模块
class BiLSTM(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob):
        super(BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout_prob = dropout_prob
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # 双向LSTM，隐藏层大小减半
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)
        # # 取最后一个时间步的输出
        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = self.dropout(lstm_out)
        # logits = self.classify(lstm_out)
        # return logits

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(last_hidden_state)
        pooled_output = h_n.squeeze(0)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return logits, loss
        else:
            return logits


# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.lstm = BiLSTM(
            self.bert.config.hidden_size,
            num_labels,
            dropout_prob
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output

        # 获取 BERT 的隐藏状态
        hidden_states = outputs.last_hidden_state
        lstm_features = self.lstm(hidden_states)

        # 合并 BERT 的池化输出和 LSTM 的特征
        pooled_output = torch.cat((pooled_output, lstm_features), dim=1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return logits, loss
        else:
            return logits

# 定义数据集类
class MyDataset(Dataset):
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

        # 拼接评论和话题内容
        input_text = f"{review} [SEP] {topic_title} {topic_text_content}"

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
            'label': torch.tensor(is_sarcasm, dtype=torch.long)
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
    plt.savefig(path)
    plt.show()
    plt.close()

# 主函数
if __name__ == '__main__':
    # 加载数据
    train_data = load_data_dev(train_path)
    topic_data = load_topic_data(train_topic_path)

    # 分割数据集为训练集和测试集
    train_data, test_data = train_test_split(train_data, test_size=test_size, train_size=train_size, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # 创建数据集
    train_dataset = MyDataset(train_data, topic_data, tokenizer)
    test_dataset = MyDataset(test_data, topic_data, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = MyModel(num_labels=2, dropout_prob=dropout_prob)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 记录每个epoch的训练损失和测试精度
    train_losses = []
    test_losses = []
    test_accuracies = []
    train_accuracies = []

    # 早停机制
    patience = patience_num
    best_accuracy = 0.0

    # 训练循环
    print("Training...")
    start_time = time.time()

    # 训练阶段
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        train_total_correct = 0  # 用于计算训练准确率
        train_total_samples = 0

        # 训练每一批次
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, labels=labels)
            logits = outputs[0]
            loss = outputs[1]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算训练准确率
            train_correct = (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total_correct += train_correct
            train_total_samples += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 计算训练准确率
        train_acc = train_total_correct / train_total_samples
        train_accuracies.append(train_acc)

        # 测试阶段
        model.eval()
        true_labels = []
        pred_labels = []
        total_test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, labels=labels)
                logits = outputs[0]
                loss = outputs[1]

                total_test_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predictions.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

        # 早停机制
        if accuracy > best_accuracy:
            patience = patience_num
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")\

    # 加载最佳模型并绘制ROC曲线
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    true_labels = []
    pred_probs = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            pred_probs.extend(probabilities[:, 1].cpu().numpy())
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, path=f'../../ROC/bi-lstm.png')

    # 计算召回率、F1分数等指标
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')

    # 输出结果
    print(f"AUC: {roc_auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")