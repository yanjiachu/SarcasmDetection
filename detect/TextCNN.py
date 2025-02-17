import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# 定义超参数
batch_size = 16
learning_rate = 5e-5
dropout_prob = 0.1
num_epochs = 3
train_size = 0.9
test_size = 0.1
train_path = '../data/train.json'
train_topic_path = '../data/train_topic.json'
model_path = '../bert-base-chinese'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


# 定义多核TextCNN模块
class TextCNN(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob):
        super(TextCNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义多个不同卷积核大小的卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=4, padding=1),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(hidden_size * 3, num_classes)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, hidden_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, hidden_size, sequence_length)

        # 应用多卷积核卷积
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        # 全局最大池化，保留最重要的特征
        out1 = torch.max(out1, dim=-1)[0]
        out2 = torch.max(out2, dim=-1)[0]
        out3 = torch.max(out3, dim=-1)[0]

        # 拼接所有卷积层的输出特征
        out = torch.cat((out1, out2, out3), dim=1)

        # 全连接层进行分类
        logits = self.fc(out)

        return logits


# 定义封装的模型
class BertTextCNNModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(BertTextCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.cnn = TextCNN(
            self.bert.config.hidden_size,
            num_labels,
            dropout_prob
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(770, num_labels)

        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)

        # 获取 BERT 的隐藏状态
        hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        cnn_features = self.cnn(hidden_states)  # (batch_size, hidden_size*3)

        # 合并 BERT 的池化输出和 CNN 的特征
        pooled_output = torch.cat((pooled_output, cnn_features), dim=1)  # (batch_size, hidden_size + 3*hidden_size)
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


# 主函数
if __name__ == '__main__':
    # 加载数据
    train_data = load_data_dev(train_path)
    topic_data = load_topic_data(train_topic_path)

    # 分割数据集为训练集和测试集
    train_data, test_data = train_test_split(train_data, test_size=test_size, train_size=train_size, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 创建数据集
    train_dataset = MyDataset(train_data, topic_data, tokenizer)
    test_dataset = MyDataset(test_data, topic_data, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = BertTextCNNModel(num_labels=2, dropout_prob=dropout_prob)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    print("Training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        # 训练每一批次
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            _, loss = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # 测试阶段
    model.eval()
    print("Evaluating...")
    with torch.no_grad():
        true_labels = []
        pred_labels = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

        # 计算准确率
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")