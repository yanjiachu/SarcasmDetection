# 中文讽刺识别

[//]: # (代码会上传至kaggle，无需配置，可直接点击运行。)

[//]: # ()
[//]: # (notebook链接:)

[//]: # ()
[//]: # ([baseline]&#40;https://www.kaggle.com/code/jiachuyan/sarcasmdetection-chinese&#41;)

[//]: # (（三个baseline）)

[//]: # ()
[//]: # ([detect]&#40;https://www.kaggle.com/code/jiachuyan/sarcasm-detect&#41;)

[//]: # (（讽刺识别任务）)

[//]: # ()
[//]: # ([classify]&#40;https://www.kaggle.com/code/jiachuyan/sarcasm-classify&#41;)

[//]: # (（讽刺分类任务）)

[//]: # ()
[//]: # ([identify]&#40;https://www.kaggle.com/code/jiachuyan/sarcasm-identify&#41;)

[//]: # (（讽刺目标识别任务）)

## 项目内容

### 1. 讽刺识别：
基于微博上下文和评论，判断目标语句是否为讽刺。
### 2. 讽刺类别识别：
结合句法特征和上下文语义信息，完成讽刺语句的类别分类任务。类别包括嘲笑、反语、讽刺文学等。
### 3. 细粒度讽刺目标识别：
采用序列标注算法，从目标语句中准确识别讽刺目标实体。

## 数据集
数据集路径：../data

### train.json
路径：../data/train.json

包含超过一万条微博评论，每条微博评论有相关联的原帖话题。

| 字段名           | 描述           | 详细说明                                   |
|---------------|--------------|----------------------------------------|
| topicId       | 评论关联的话题id    | 相关话题内容存放于train_topic.json              |
| review        | 评论内容         | 清洗过的中文评论                               |
| isSarcasm     | 该评论是否为讽刺     | 1代表讽刺，0代表非讽刺                           |
| sarcasmType   | 该评论的讽刺类型     | 若isSarcasm字段为0，则该字段为null；否则1~6表示六种讽刺类型 |
| dataId        | 评论id         |                                        |
| sarcasmTarget | 该评论的讽刺目标实体列表 | 目标实体可能有多个                              |

示例：
```json
{   
    "topicId": 13, 
    "review": "被扎的那个得是过命的交情", 
    "isSarcasm": 1, 
    "sarcasmType": 3, 
    "dataId": 8485, 
    "sarcasmTarget": ["医学生"]
}
```

### train_topic.json
路径：../data/train_topic.json

包含40个微博话题，每条都包含话题的标题和内容。

| 字段名          | 描述   | 详细说明 |
|--------------|------|------|
| topicId      | 话题id |      |
| topicTitle   | 话题标题 |      |
| topicContent | 话题内容 |      |

示例：
```json
{
    "topicId": 13, 
    "topicTitle": "医学生第一次给人打针，眼睛一闭", 
    "topicContent": "#医学生第一次给人打针#哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈“眼睛一闭”可还行。"
}
```


## 讽刺识别

| 预训练模型        | 下游任务模型  | 验证准确率(%) |
|--------------|---------|----------|
| Bert冻结全部参数   | Linear  | 69.19    |
| Bert冻结全部参数   | MLP     | 69.32    |
| Bert冻结全部参数   | TextCNN | 78.01    |
| Bert冻结全部参数   | GRU     | 72.29    |
| Bert冻结全部参数   | LSTM    | 72.58    |
| Bert冻结前10层参数 | Linear  | 76.19    |
| Bert冻结前10层参数 | TextCNN | 77.92    |
| Bert冻结前9层参数  | Linear  | 75.55    |

## 讽刺类别识别

| 预训练模型      | 下游任务模型  | 验证准确率(%) |
|------------|---------|----------|
| Bert冻结全部参数 | Linear  | 63.90    |
| Bert冻结全部参数 | MLP     | 63.32    |
| Bert冻结全部参数 | TextCNN | 77.96    |
| Bert冻结全部参数 | GRU     | 69.52    |
| Bert冻结全部参数 | LSTM    | 67.99    |

## 讽刺目标识别

| 预训练模型      | 下游任务模型  | 验证准确率(%) |
|------------|---------|----------|
| Bert冻结全部参数 | Linear  | 17.51    |
| Bert冻结全部参数 | MLP     | 28.88    |
| Bert冻结全部参数 | TextCNN | 38.40    |
| Bert冻结全部参数 | GRU     | 41.60    |
| Bert冻结全部参数 | LSTM    | 42.56    |
