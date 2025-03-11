import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# 定义超参数
batch_size = 32
learning_rate = 1e-3
dropout_prob = 0.05
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
best_model_path = '../../models/detect/bi-lstm.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, topic_dict, tfidf_vectorizer):
        self.data = data
        self.topic_dict = topic_dict
        self.tfidf_vectorizer = tfidf_vectorizer

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
        # input_text = f"{review} {topic_text_content}"
        # input_text = f"{review} {topic_title}"
        input_text = f"{review} {topic_title} {topic_text_content}"

        # 使用 TF-IDF 向量化
        input_vector = self.tfidf_vectorizer.transform([input_text]).toarray()[0]
        input_vector = torch.tensor(input_vector, dtype=torch.float32)
        label = torch.tensor(is_sarcasm, dtype=torch.long)

        return {
            'input_vector': input_vector,  # TF-IDF 向量
            'label': label  # 标签
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

# 训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in dataloader:
        input_vector = batch['input_vector'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(input_vector)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算指标
        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

# 验证函数
def eval(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_vector = batch['input_vector'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(input_vector)
            loss = criterion(outputs, labels)

            # 计算指标
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

if __name__ == '__main__':
    # 加载数据
    data = load_data_dev(train_path)
    topic_dict = load_topic_data(train_topic_path)

    # 划分训练集和验证集
    train_data, test_data = train_test_split(data, test_size=test_size, train_size=train_size, random_state=42)

    # 初始化 TF-IDF 向量化器
    tfidf_vectorizer = TfidfVectorizer(max_features=4096)

    # 准备 TF-IDF 向量化器的训练数据
    texts = []
    for item in data:
        topic_id = item['topicId']
        review = item['review']
        topic_content = topic_dict.get(topic_id, {})
        topic_title = topic_content.get('topicTitle', '')
        topic_text_content = topic_content.get('topicContent', '')
        input_text = f"{review} {topic_title}"
        texts.append(input_text)

    # 拟合 TF-IDF 向量化器
    tfidf_vectorizer.fit(texts)

    # 初始化数据集
    train_dataset = MyDataset(train_data, topic_dict, tfidf_vectorizer)
    test_dataset = MyDataset(test_data, topic_dict, tfidf_vectorizer)

    # 初始化 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTM(input_size=4096, hidden_size=256, num_classes=2,dropout_prob=dropout_prob).to(device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证
    best_test_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        avg_train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        avg_test_loss, test_accuracy = eval(model, test_loader, criterion)
        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy * 100:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Acc: {test_accuracy * 100:.2f}%")

        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)

    # 加载最佳模型并绘制ROC曲线
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    true_labels = []
    pred_probs = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_vector = batch['input_vector'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_vector)
            probabilities = torch.softmax(logits, dim=1)
            pred_probs.extend(probabilities[:, 1].cpu().numpy())
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    # 计算召回率、F1分数等指标
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    print(f"AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")