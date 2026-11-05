# 02 · Token 与向量

## 一句话总结

文字先被切成 **token**（介于字和词之间的碎片），每个 token 变成一个**向量**（一串数字），模型的全部运算都发生在这堆数字上。

## 核心概念

### 1. Token 不是字，也不是词

中文"人工智能"可能被切成 `人工` + `智能`，也可能 `人` + `工` + `智能`，取决于分词表。英文 `unbelievable` 常被切成 `un` + `believ` + `able`。

三条实用推论：

| 现象 | 原因 |
|---|---|
| 中文往往比英文贵 | 同样意思，中文的 token 数常是英文的 1.5–2 倍 |
| 模型数不清"这句话有几个字" | 它看见的是 token，不是字符 |
| 生僻字/特殊符号表现差 | 切碎了，训练里见到的次数少 |

### 2. 向量：把"意思"变成坐标

每个 token 对应一个几百到几千维的向量。距离近 = 用法相近。

经典玩具例子（二维示意）：

```
猫  → (0.9, 0.85)
狗  → (0.88, 0.80)
汽车 → (-0.7, 0.2)
```

"猫"和"狗"方向接近，"汽车"差很远。真实向量维度高得多，但几何直觉一样。

由此还能做算术：`国王 - 男人 + 女人 ≈ 女王`。这说明向量空间里编码了**关系**，不只是相似度。

### 3. 余弦相似度

判断两段文本像不像，最常用的是夹角余弦：

```
cos(A,B) = (A·B) / (|A|·|B|)   取值 -1 ~ 1，越大越像
```

这就是**向量检索**的全部基础：把文档和问题都变成向量，算相似度，取最高的几条塞进上下文。

### 4. 上下文窗口

模型一次能看的 token 上限（比如 128k）。注意三个坑：

1. **不是越大越好**：注意力计算量随长度平方增长，长上下文更慢更贵
2. **中间会丢**：模型对开头和结尾更敏感，正中间的内容容易被忽略（"lost in the middle"）
3. **满了会截断**：老对话被挤掉，它就真的忘了

## 动手实验

### 实验 1：数 token

```python
import tiktoken  # pip install tiktoken
enc = tiktoken.get_encoding("cl100k_base")
for s in ["人工智能正在改变世界", "AI is changing the world", "def fib(n): return n"]:
    toks = enc.encode(s)
    print(len(toks), toks, [enc.decode([t]) for t in toks])
```

**观察**：中文一句话多少 token？英文呢？代码呢？

### 实验 2：算相似度

```python
from openai import OpenAI
client = OpenAI()

def embed(t):
    return client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding

import math
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

pairs = [
    ("怎么退款", "我要退钱"),
    ("怎么退款", "今天天气不错"),
    ("Python 列表", "Python 数组"),
]
for a, b in pairs:
    print(f"{cos(embed(a), embed(b)):.3f}  {a} / {b}")
```

**观察**：语义相近但字面不同的两句，相似度明显更高。这就是 RAG 能work的原因。

### 实验 3：上下文遗忘

连续聊 20 轮，第 1 轮告诉它"我叫小张，住在杭州"，第 20 轮问"我叫什么，住哪"。再把它塞满无关内容重试。

## 自测三问

1. 为什么"模型数不清字数"是 token 化导致的？
2. 两个句子字面完全不像，但向量很近，这说明向量编码了什么？
3. 上下文窗口 128k，为什么把一本 300 页的书整本塞进去效果反而差？

## 延伸阅读

- OpenAI《Tokenizer》在线工具（可视化切分）
- 《Lost in the Middle》论文（长上下文中间遗忘）
- 《Efficient Estimation of Word Representations》Word2Vec 原论文
