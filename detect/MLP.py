import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# 定义超参数
batch_size = 16
learning_rate = 5e-5
dropout_prob = 0
patience_num = 6    # 早停阈值
draw_step = 3       # 绘制loss和acc的图像的间隔，建议与早停机制配合
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../data/train.json'
train_topic_path = '../data/train_topic.json'
model_path = '../bert-base-chinese'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.fc1 = torch.nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = torch.nn.Linear(256, 32)
        self.classifier = torch.nn.Linear(32, num_labels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)

        # 冻结 BERT 参数，除了最后1层
        # for name, param in self.bert.named_parameters():
        #     if 'encoder.layer.11' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # 冻结 BERT 参数，除了最后2层
        # for name, param in self.bert.named_parameters():
        #     if 'encoder.layer.11' in name or 'encoder.layer.10' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # 冻结 BERT 所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output

        pooled_output = self.fc1(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)

        pooled_output = self.fc2(pooled_output)
        pooled_output = self.relu(pooled_output)
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
        train_accuracy = train_total_correct / train_total_samples
        train_accuracies.append(train_accuracy)

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

        test_accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy * 100:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Acc: {test_accuracy * 100:.2f}%")
        # 写入日志
        with open(f'../logs/detect/1_none_MLP_{num_epochs}.txt', 'a') as f:
            f.write(f"Epoch {epoch}/{num_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_accuracy * 100:.2f}%, "
                    f"Test Loss: {avg_test_loss:.4f}, "
                    f"Test Acc: {test_accuracy * 100:.2f}%\n")

        # 阶段输出图像
        if epoch % draw_step == 0:
            plot_loss_acc(train_losses, test_losses, train_accuracies, test_accuracies, epoch,
                path=f'../training_curves/detect/1_none_MLP_{epoch}.png'
            )

        # 早停机制
        if test_accuracy > best_accuracy:
            patience = patience_num
            best_accuracy = test_accuracy
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                with open(f'../logs/detect/1_none_MLP_{num_epochs}.txt', 'a') as f:
                    f.write("Early stopping!\n")
                break

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    with open(f'../logs/detect/1_none_MLP_{num_epochs}.txt', 'a') as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")