# 中文讽刺识别

## 项目内容

### 1. 讽刺识别：
基于微博上下文和评论，判断目标语句是否为讽刺。
### 2. 讽刺类别识别：
结合句法特征和上下文语义信息，完成讽刺语句的类别分类任务。类别包括嘲笑、反语、讽刺文学等。
### 3. 细粒度讽刺目标识别：
采用序列标注算法，从目标语句中准确识别讽刺目标实体。

## 数据集

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

| 模型     | AUC    | Recall | F1 Score |
|---------|--------|--------|----------|
| BERT    | 0.8581 | 0.8753 | 0.8527   |
| model_1 | 0.8369 | 0.8614 | 0.8383   |
| model_2 | 0.8304 | 0.8576 | 0.8377   |

### BERT

{评论}[SEP]{标题}{内容}输入bert处理后直接通过分类层输出

AUC: 0.8581

Recall: 0.8753

F1 Score: 0.8527

### model_1:

评论单独过bert，{评论}[SEP]{标题}{内容}过bert，二者拼接经过一层线性层，再通过分类层输出

AUC: 0.8369

Recall: 0.8614

F1 Score: 0.8383

### model_2:

#### bert
评论单独过bert，{标题}[SEP]{内容}过bert，二者拼接经过一层线性层，再通过分类层输出

AUC: 0.8304

Recall: 0.8576

F1 Score: 0.8377

#### macbert

评论单独过macbert，{标题}[SEP]{内容}过macbert，二者拼接经过一层线性层，再通过分类层输出

AUC: 0.8238

Recall: 0.9101

F1 Score: 0.8404

#### lert

评论单独过lert，{标题}[SEP]{内容}过lert，二者拼接经过一层线性层，再通过分类层输出

AUC: 0.8529

Recall: 0.8861

F1 Score: 0.8508

### model_3:

评论单独过bert，内容过bert，二者拼接经过一层线性层，再通过分类层输出

AUC: 0.8338

Recall: 0.8975

F1 Score: 0.8440

### model_4:

分别经过bert+CNN，最后池化经过分类层输出

## 讽刺类别识别

### 数据集说明

标记为讽刺的评论分为6种类型，详细描述如下表所示：

| 标签 | 类型   | 解释               | 示例                 |
|----|------|------------------|--------------------|
| 1  | 直接讽刺 | 直接点明讽刺对象         | “想涨价就直说”           |
| 2  | 夸张讽刺 | 通过夸张表达讽刺         | “我再涨涨你就知道现在有多便宜了”  |
| 3  | 反话讽刺 | 表面上说正面的话，实际上表达讽刺 | “支持啊！公交车能不能也加入啊”   |
| 4  | 荒诞讽刺 | 通过极端荒诞的假设来讽刺     | “把人都枪毙了岂不更省事儿？”    |
| 5  | 自嘲讽刺 | 讽刺中带有自嘲的意味       | “好嘛，上学的活不起了”       |
| 6  | 质疑讽刺 | 通过反问或质疑表达讽刺      | “目的是错峰？给涨价找个借口罢了！” |

其中，标签1有9754条评论，标签2有1460条评论，标签3有1064条评论，标签4有55条评论，标签5有175条评论，标签6有3135条评论。共计11209条评论。

## 讽刺目标识别

