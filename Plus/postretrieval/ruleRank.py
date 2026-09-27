# 重排   --->  基于规则的重排方案
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


from Plus import util, config
# 指定数据集
DATA_PATH = config.POST_RETRIEVAL_DATA_PATH / "reRank.json"

# 记录重新排序之后需要几条数据
TOP_K = 5

# json数据的结果
records = util.load_records(DATA_PATH)

# 文档列表
documents = []

# 嵌入模型
embed_model = util.get_embed_model()

# 遍历json数组，构建Document列表
for record in records:
    text = f"书名：{record['book_title']}， 摘要：{record['book_content']}"
    metadata = {
        "book_id": record["book_id"],
        "book_title": record["book_title"],
        "book_content": record["book_content"],
    }
    documents.append(Document(text=text, metadata=metadata))

# 构建向量索引
index = VectorStoreIndex.from_documents(documents, embed_model= embed_model)

# 创建检索器
retriever = VectorIndexRetriever(index=index, similarity_top_k=len(records))

# 模拟同学找书的问题
query = "高三高考备考资料，物理电磁学相关的冲刺习题"

raw_nodes : list[NodeWithScore] = retriever.retrieve(query)

print(f"初步检索结果，共{len(raw_nodes)}条")
for node in raw_nodes:
    print(f"{node.metadata['book_id']} | 分数：{node.score:.2f} |{node.metadata['book_title']}")

print("-" * 60)

# 规则重排的方法
def rule_score(title: str, content: str) -> float:
    text = f"{title}{content}"

    # 打分的规则
    score = 10.0 if any(k in text for k in ("电磁", "电场", "磁场", "电磁感应"))else 0.0

    if any(k in text for k in ("高考", "高三", "冲刺")):
        score += 8

    if any(k in text for k in ("大题", "压轴", "真题")):
        score += 6

    if any(k in text for k in ("力学", "光学", "声学", "初中")):
        score -= 10

    if any(k in text for k in ("无电磁学", "不涉及电磁场")):
        score -= 20

    return score

for node in raw_nodes:
    node.score += rule_score(node.metadata["book_title"],node.metadata["book_content"])

# 按照重新打分的数值，进行重排
reranked_nodes = sorted(raw_nodes, key=lambda  x: x.score, reverse=True)[:TOP_K]

# 输出基于规则的重排内容
for i, node in enumerate(reranked_nodes, 1):
    print(
        f"[{i}] {node.metadata['book_id']} | 规则得分：{node.score:.2f} | "
        f"{node.metadata['book_title']} | {node.metadata['book_content']}"
    )

print("-" * 60)

print("【基于交叉编码器重排】")

# 使用本地 Ollama Rerank 模型重新计算相关性分数
for node in raw_nodes:
    score = util.rerank_score(
        query,
        node.get_content()
    )

    # 用重排分数覆盖原来的向量相似度分数
    node.score = score

# 按照重排分数降序排列，取前 TOP_K
reranked_nodes = sorted(raw_nodes,key=lambda x: x.score,reverse=True)[:TOP_K]

# 输出基于 Reranker 的重排结果
for i, node in enumerate(reranked_nodes, 1):
    print(
        f"[{i}] {node.metadata['book_id']} | 重排得分：{node.score:.4f} | "
        f"{node.metadata['book_title']} | {node.metadata['book_content']}"
    )