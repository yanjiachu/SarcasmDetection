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

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob, hidden_size=768):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.gru = torch.nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state

        # 提取最后一层的隐藏状态
        gru_output, h_n = self.gru(last_hidden_state)

        # 取最后一步的隐藏状态作为特征
        pooled_output = h_n.squeeze(0)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
    tokenizer = BertTokenizer.from_pretrained(model_path)

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
    model = MyModel(num_labels=6, dropout_prob=dropout_prob)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    print("Training...")
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0

        # 训练每一批次
        for batch in train_loader:
            # 合并批次数据
            input_ids = []
            attention_masks = []
            labels = []
            for sample in batch:
                if sample is not None:
                    input_ids.append(sample['input_ids'])
                    attention_masks.append(sample['attention_mask'])
                    labels.append(sample['label'])

            if not input_ids:
                continue

            input_ids = torch.stack(input_ids).to(device)
            attention_masks = torch.stack(attention_masks).to(device)
            labels = torch.stack(labels).to(device)

            optimizer.zero_grad()

            _, loss = model(input_ids, attention_masks, labels=labels)
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
            input_ids = []
            attention_masks = []
            labels = []
            for sample in batch:
                if sample is not None:
                    input_ids.append(sample['input_ids'])
                    attention_masks.append(sample['attention_mask'])
                    labels.append(sample['label'])

            if not input_ids:
                continue

            input_ids = torch.stack(input_ids).to(device)
            attention_masks = torch.stack(attention_masks).to(device)
            labels = torch.stack(labels).to(device)

            logits = model(input_ids, attention_masks)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

        # 计算准确率
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")