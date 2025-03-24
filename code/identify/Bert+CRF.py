import json
import torch
import time
import numpy as np
from pytorchcrf import CRF
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split

# 定义超参数
batch_size = 32
learning_rate = 2e-5
dropout_prob = 0.1
patience_num = 3  # 早停阈值
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
model_path = '../../bert-base-chinese'
best_model_path = '../../models/identify/CRF.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        # 冻结 BERT 参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state
        pooled_output = self.dropout(pooled_output)
        emissions = self.classifier(pooled_output)

        if labels is not None:
            # 计算 CRF 的负对数似然损失
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return emissions, loss
        else:
            return emissions

    def predict(self, input_ids, attention_mask):
        emissions = self.forward(input_ids, attention_mask)
        # 使用 CRF 的解码方法获取最优路径
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        return predictions


# 定义标签
label2id = {'B-ORG': 0, 'I-ORG': 1, 'O': 2}

# 定义数据集类
class SarcasmTargetDataset(Dataset):
    def __init__(self, data, topic_dict, tokenizer, label2id):
        self.data = data
        self.topic_dict = topic_dict
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        topic_id = item['topicId']
        review = item['review']
        is_sarcasm = item['isSarcasm']
        sarcasm_target = item['sarcasmTarget']

        # 只处理标签为1且存在sarcasmTarget的讽刺文本
        if is_sarcasm == 1 and sarcasm_target:
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
                return_offsets_mapping=True,  # 获取子词映射
                return_tensors='pt'
            )

            # 生成标签
            offsets = encoding['offset_mapping'].squeeze().tolist()
            labels = [self.label2id['O']] * len(offsets)
            # 只对 review 部分进行标注
            review_length = len(review)
            for target in sarcasm_target:
                if not isinstance(target, str):  # 确保目标是字符串类型
                    continue  # 跳过非字符串类型的目标

                # 找到目标词在 review 中的位置
                target_start = review.find(target)
                if target_start == -1:
                    continue  # 目标词不在 review 中，跳过

                target_end = target_start + len(target)

                # 遍历 offsets，找到目标词对应的 token 位置
                for i, (start, end) in enumerate(offsets):
                    if start >= review_length:
                        break  # 超出 review 部分，停止标注

                    # 如果当前 token 在目标词的范围内
                    if start <= target_start < end:
                        labels[i] = self.label2id['B-ORG']
                    elif start < target_end <= end:
                        labels[i] = self.label2id['I-ORG']

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:
            return None  # 跳过非讽刺或无sarcasmTarget的样本

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

    # 筛选出有效数据
    filtered_data = [item for item in train_data if item['isSarcasm'] == 1 and item['sarcasmTarget']]

    # 分割数据集为训练集和测试集
    train_data, test_data = train_test_split(filtered_data, test_size=test_size, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_path, use_fast=True)

    # 创建数据集
    train_dataset = SarcasmTargetDataset(train_data, topic_data, tokenizer, label2id)
    test_dataset = SarcasmTargetDataset(test_data, topic_data, tokenizer, label2id)

    # 过滤掉无效的样本
    train_dataset.data = [item for item in train_dataset.data if item is not None]
    test_dataset.data = [item for item in test_dataset.data if item is not None]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = MyModel(num_labels=len(label2id), dropout_prob=dropout_prob)
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
    best_loss = float('inf')  # 初始化为无穷大

    # 训练循环
    print("Training...")
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0
        total_correct_comments = 0
        total_comments = 0

        # 训练每一批次
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 获取 logits 和 loss
            logits, loss = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算评论级别的训练精度
            predictions = torch.argmax(logits, dim=2)
            batch_true_labels = labels.cpu().numpy()
            batch_pred_labels = predictions.cpu().numpy()

            # 逐条评论检查是否所有token都预测正确
            for i in range(len(batch_true_labels)):
                is_correct = np.all(batch_true_labels[i] == batch_pred_labels[i])
                total_correct_comments += int(is_correct)
                total_comments += 1

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 计算并保存评论级别的训练精度
        train_accuracy = total_correct_comments / total_comments
        train_accuracies.append(train_accuracy)

        # 测试阶段
        model.eval()
        true_labels = []
        pred_labels = []
        comment_correctness = []
        total_test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 获取 logits 和 loss
                logits, test_loss = model(input_ids, attention_mask, labels=labels)
                predictions = torch.argmax(logits, dim=2)

                # 累加测试损失
                total_test_loss += test_loss.item()

                # 将预测结果和真实标签转换为CPU和numpy格式
                batch_true_labels = labels.cpu().numpy()
                batch_pred_labels = predictions.cpu().numpy()

                # 逐条评论检查是否所有token都预测正确
                for i in range(len(batch_true_labels)):
                    is_correct = np.all(batch_true_labels[i] == batch_pred_labels[i])
                    comment_correctness.append(is_correct)

                # 保存token级别的结果
                true_labels.extend(batch_true_labels)
                pred_labels.extend(batch_pred_labels)

        # 计算评论级别的准确率
        comment_accuracy = np.mean(comment_correctness)
        test_accuracies.append(comment_accuracy)

        # 计算并保存测试损失
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # 打印结果
        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy * 100:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Acc: {comment_accuracy * 100:.2f}%")

        # 早停机制
        if avg_test_loss < best_loss:
            patience = patience_num
            best_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
