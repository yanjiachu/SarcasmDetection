import json
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast
from sklearn.model_selection import train_test_split

# 定义超参数
batch_size = 32
learning_rate = 2e-5
dropout_prob = 0.1
patience_num = 3    # 早停阈值
num_epochs = 30
train_size = 0.9
test_size = 0.1
train_path = '../../data/train.json'
train_topic_path = '../../data/train_topic.json'
model_path = '../../bert-base-chinese'
# model_path = '../../chinese-lert-base'
best_model_path = '../../models/identify/bert.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 定义封装的模型
class MyModel(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

        # 冻结 BERT 参数
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return logits, loss
        else:
            return logits


# 定义标签
label2id = {'B-ORG': 0, 'I-ORG': 1, 'O': 2}


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
        sarcasm_type = item['sarcasmType']
        sarcasm_target = item['sarcasmTarget']

        if is_sarcasm == 1 and sarcasm_target:
            topic_content = self.topic_dict.get(topic_id, {})
            topic_title = topic_content.get('topicTitle', '')
            topic_text_content = topic_content.get('topicContent', '')

            # 使用tokenizer的sep_token拼接
            input_text = f"{review}{self.tokenizer.sep_token}{topic_title}{topic_text_content}"

            encoding = self.tokenizer(
                input_text,
                padding='max_length',
                max_length=256,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            offsets = encoding['offset_mapping'].squeeze().tolist()
            labels = [self.label2id['O']] * len(offsets)
            valid_targets = []

            # 预处理：确保target为字符串并去重
            for target in sarcasm_target:
                if isinstance(target, str) and target not in valid_targets:
                    valid_targets.append(target)

            # 仅处理review中的目标
            for target in valid_targets:
                start_char = review.find(target)
                if start_char == -1:
                    continue  # 目标不在review中
                end_char = start_char + len(target)

                # 查找对应的token索引
                start_token_idx = None
                end_token_idx = None

                # 查找起始token
                for i, (token_start, token_end) in enumerate(offsets):
                    if token_start <= start_char < token_end:
                        start_token_idx = i
                        break
                if start_token_idx is None:
                    continue

                # 查找结束token
                for i in range(start_token_idx, len(offsets)):
                    token_start, token_end = offsets[i]
                    if token_end >= end_char:
                        end_token_idx = i
                        break
                if end_token_idx is None:
                    end_token_idx = len(offsets) - 1

                # 标注B-和I-标签
                labels[start_token_idx] = self.label2id['B-ORG']
                for j in range(start_token_idx + 1, end_token_idx + 1):
                    if j < len(labels):
                        labels[j] = self.label2id['I-ORG']

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:
            return None

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

def compute_exact_match_f1(true_labels, pred_labels, offsets_mapping, original_texts, label2id):
    id2label = {v: k for k, v in label2id.items()}
    tp, fp, fn = 0, 0, 0

    for i in range(len(true_labels)):
        true = true_labels[i]
        pred = pred_labels[i]
        offsets = offsets_mapping[i]
        text = original_texts[i]

        # 提取真实和预测的Span（B-ORG/I-ORG连续的区间）
        def extract_spans(labels):
            spans = []
            current_span = []
            for j, label_id in enumerate(labels):
                label = id2label.get(label_id, 'O')
                if label == 'B-ORG':
                    if current_span:
                        spans.append(current_span)
                    current_span = [j]
                elif label == 'I-ORG' and current_span:
                    current_span.append(j)
                else:
                    if current_span:
                        spans.append(current_span)
                    current_span = []
            if current_span:
                spans.append(current_span)
            return spans

        true_spans = extract_spans(true)
        pred_spans = extract_spans(pred)

        # 将Span转换为文本（严格匹配）
        def get_span_text(span, offsets, text):
            if not span:
                return ""
            start = offsets[span[0]][0]
            end = offsets[span[-1]][1]
            return text[start:end]

        # 统计TP/FP/FN
        matched_true = set()
        matched_pred = set()

        # 检查预测的Span是否与真实Span完全匹配
        for pred_span in pred_spans:
            pred_text = get_span_text(pred_span, offsets, text)
            is_match = False
            for true_span in true_spans:
                true_text = get_span_text(true_span, offsets, text)
                if pred_text == true_text:
                    tp += 1
                    matched_true.add(tuple(true_span))
                    matched_pred.add(tuple(pred_span))
                    is_match = True
                    break
            if not is_match:
                fp += 1

        # 未被匹配的真实Span是FN
        for true_span in true_spans:
            if tuple(true_span) not in matched_true:
                fn += 1

    # 计算Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Exact Match Precision": precision,
        "Exact Match Recall": recall,
        "Exact Match F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }


def compute_dice_score(true_labels, pred_labels, label2id):
    id2label = {v: k for k, v in label2id.items()}

    # Extract unique predicted and true spans
    def extract_unique_spans(labels):
        spans = set()
        current_span = []
        for label_id in labels:
            label = id2label.get(label_id, 'O')
            if label == 'B-ORG':
                if current_span:
                    spans.add(tuple(current_span))
                current_span = [label_id]
            elif label == 'I-ORG' and current_span:
                current_span.append(label_id)
            else:
                if current_span:
                    spans.add(tuple(current_span))
                current_span = []
        if current_span:
            spans.add(tuple(current_span))
        return spans

    total_true_unique = 0
    total_pred_unique = 0
    total_common = 0

    for true, pred in zip(true_labels, pred_labels):
        true_spans = extract_unique_spans(true)
        pred_spans = extract_unique_spans(pred)

        total_true_unique += len(true_spans)
        total_pred_unique += len(pred_spans)
        total_common += len(true_spans & pred_spans)

    # Avoid division by zero
    if total_true_unique + total_pred_unique == 0:
        return 0.0

    dice_score = 2 * total_common / (total_true_unique + total_pred_unique)
    return dice_score

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
    best_loss = float('inf')

    # # 训练循环
    # print("Training...")
    # start_time = time.time()
    # for epoch in range(1, num_epochs + 1):
    #     model.train()  # 设置模型为训练模式
    #     total_loss = 0.0
    #     total_correct_comments = 0
    #     total_comments = 0
    #
    #     # 训练每一批次
    #     for batch in train_loader:
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #
    #         optimizer.zero_grad()
    #
    #         # 获取 logits 和 loss
    #         logits, loss = model(input_ids, attention_mask, labels=labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #         # 计算评论级别的训练精度
    #         predictions = torch.argmax(logits, dim=2)
    #         batch_true_labels = labels.cpu().numpy()
    #         batch_pred_labels = predictions.cpu().numpy()
    #
    #         # 逐条评论检查是否所有token都预测正确
    #         for i in range(len(batch_true_labels)):
    #             is_correct = np.all(batch_true_labels[i] == batch_pred_labels[i])
    #             total_correct_comments += int(is_correct)
    #             total_comments += 1
    #
    #     avg_train_loss = total_loss / len(train_loader)
    #     train_losses.append(avg_train_loss)
    #
    #     # 计算并保存评论级别的训练精度
    #     train_accuracy = total_correct_comments / total_comments
    #     train_accuracies.append(train_accuracy)
    #
    #     # 测试阶段
    #     model.eval()
    #     true_labels = []
    #     pred_labels = []
    #     comment_correctness = []
    #     total_test_loss = 0.0
    #
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             input_ids = batch['input_ids'].to(device)
    #             attention_mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)
    #
    #             # 获取 logits 和 loss
    #             logits, test_loss = model(input_ids, attention_mask, labels=labels)
    #             predictions = torch.argmax(logits, dim=2)
    #
    #             # 累加测试损失
    #             total_test_loss += test_loss.item()
    #
    #             # 将预测结果和真实标签转换为CPU和numpy格式
    #             batch_true_labels = labels.cpu().numpy()
    #             batch_pred_labels = predictions.cpu().numpy()
    #
    #             # 逐条评论检查是否所有token都预测正确
    #             for i in range(len(batch_true_labels)):
    #                 is_correct = np.all(batch_true_labels[i] == batch_pred_labels[i])
    #                 comment_correctness.append(is_correct)
    #
    #             # 保存token级别的结果
    #             true_labels.extend(batch_true_labels)
    #             pred_labels.extend(batch_pred_labels)
    #
    #     # 计算评论级别的准确率
    #     comment_accuracy = np.mean(comment_correctness)
    #     test_accuracies.append(comment_accuracy)
    #
    #     # 计算并保存测试损失
    #     avg_test_loss = total_test_loss / len(test_loader)
    #     test_losses.append(avg_test_loss)
    #
    #     # 打印结果
    #     print(f"Epoch {epoch}/{num_epochs}, "
    #           f"Train Loss: {avg_train_loss:.4f}, "
    #           f"Train Acc: {train_accuracy * 100:.2f}%, "
    #           f"Test Loss: {avg_test_loss:.4f}, "
    #           f"Test Acc: {comment_accuracy * 100:.2f}%")
    #
    #     # 早停机制
    #     if avg_test_loss < best_loss:
    #         patience = patience_num
    #         best_loss = avg_test_loss
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

    # 重新处理测试集以获取原始文本和offset_mapping
    original_texts = []
    true_labels = []
    pred_labels = []
    offsets_mapping = []

    for item in test_data:
        if item['isSarcasm'] == 1 and item['sarcasmTarget']:
            topic_content = topic_data.get(item['topicId'], {})
            topic_title = topic_content.get('topicTitle', '')
            topic_text_content = topic_content.get('topicContent', '')
            input_text = f"{item['review']}{tokenizer.sep_token}{topic_title}{topic_text_content}"

            encoding = tokenizer(
                input_text,
                padding='max_length',
                max_length=256,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            # 获取真实标签（与Dataset中相同的逻辑）
            offsets = encoding['offset_mapping'].squeeze().tolist()
            labels = [label2id['O']] * len(offsets)
            for target in item['sarcasmTarget']:
                if isinstance(target, str):
                    start_char = item['review'].find(target)
                    if start_char != -1:
                        end_char = start_char + len(target)
                        start_token_idx = None
                        for i, (token_start, token_end) in enumerate(offsets):
                            if token_start <= start_char < token_end:
                                start_token_idx = i
                                break
                        if start_token_idx is not None:
                            end_token_idx = None
                            for i in range(start_token_idx, len(offsets)):
                                token_start, token_end = offsets[i]
                                if token_end >= end_char:
                                    end_token_idx = i
                                    break
                            if end_token_idx is None:
                                end_token_idx = len(offsets) - 1
                            labels[start_token_idx] = label2id['B-ORG']
                            for j in range(start_token_idx + 1, end_token_idx + 1):
                                if j < len(labels):
                                    labels[j] = label2id['I-ORG']

            # 模型预测
            with torch.no_grad():
                logits = model(
                    input_ids=encoding['input_ids'].to(device),
                    attention_mask=encoding['attention_mask'].to(device)
                )
                predictions = torch.argmax(logits, dim=2).squeeze().cpu().numpy()

            # 保存结果
            original_texts.append(item['review'])
            true_labels.append(labels)
            pred_labels.append(predictions)
            offsets_mapping.append(offsets)

    # 计算Exact Match F1
    metrics = compute_exact_match_f1(true_labels, pred_labels, offsets_mapping, original_texts, label2id)
    print(f"Precision: {metrics['Exact Match Precision']:.4f}")
    print(f"Recall:    {metrics['Exact Match Recall']:.4f}")
    print(f"F1:        {metrics['Exact Match F1']:.4f}")

    # 计算DICE
    dice_score = compute_dice_score(true_labels, pred_labels, label2id)
    print(f"DICE Score: {dice_score:.4f}")