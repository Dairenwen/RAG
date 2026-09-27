# 检索后处理   ---->    结果去重
import hashlib
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from Plus import util, config


# 语义去重的相似度阈值
SEMANTIC_SIM = 0.85

records = util.load_records(config.POST_RETRIEVAL_DATA_PATH / "repeat.json")

documents = []

# 嵌入模型
embed_model = util.get_embed_model()

# 把json数组转换成Document列表
for item in records:
    metadata = {
        "record_id": item["id"],
        "text": item["text"],
    }
    documents.append(Document(text=item["text"], metadata=metadata))

# 构建向量索引
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 创建检索器
retriever = VectorIndexRetriever(index=index, similarity_top_k=len(records))

# 模拟同学提问
query = "请假超时会有什么处罚？"
raw_nodes : list[NodeWithScore] = retriever.retrieve(query)

print(f"初步检索结果，共{len(raw_nodes)}条")

for n in raw_nodes:
    print(f"{n.metadata['record_id']} | 分数：{n.score:.2f} |{n.metadata['text']}")

print("-" * 60)


print("【第一层】，实现精确去重")

# 原始文本转md5
def _exact_key(node : NodeWithScore) -> str:
    # 先取出正文
    text = node.metadata.get("text")
    normalized = " ".join(text.split())

    # 实现了把数据集正文转换成32位字符串
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

# 字典：key是去重键，value是当前key下面得分最高的结果果
best : dict[str, NodeWithScore] = {}
order : list[str] =[] # 保存

# 遍历查询到的结果
for node in raw_nodes:
    key = _exact_key(node)
    if key not in best:
        order.append(key)
        best[key] = node
    elif node.score > best[key].score:
        best[key] = node

nodes_after_exact = [best[k] for k in order ]
print(f"精确去重后剩余 {len(nodes_after_exact)}条")

for n in nodes_after_exact:
    print(f"{n.metadata['record_id']} | 分数：{n.score:.2f} |{n.metadata['text']}")

print("-" * 60)


print("【第二层】，实现语义去重")
kept_nodes: list[NodeWithScore] = []
kept_embeddings : list[list[float]] = [] # 保存已经保留的节点的embedding


for node in sorted(nodes_after_exact, key=lambda x: x.score, reverse=True):
    embedding = embed_model.get_text_embedding(node.metadata.get("text"))

    is_duplicate = any(

        embed_model.similarity(embedding, kept_emb) >= SEMANTIC_SIM for kept_emb in kept_embeddings
    )

    if not is_duplicate:
        kept_nodes.append(node)
        kept_embeddings.append(embedding)

kept_ids = {n.metadata["record_id"] for n in kept_nodes}
nodes_after_semantic = [n for n in nodes_after_exact if n.metadata["record_id"] in kept_ids ]

print(f"语义去重后剩余：{len(nodes_after_semantic)}条")

for n in nodes_after_semantic:
    print(f"{n.metadata['record_id']} | 分数：{n.score:.2f} |{n.metadata['text']}")