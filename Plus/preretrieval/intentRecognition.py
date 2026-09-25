from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from Plus import util,config

# 1. 构建意图节点列表
def build_intent_nodes(records : list[dict]) -> list[TextNode]:

    # 1. 定义空列表，将来存放生成的意图节点
    nodes : list[TextNode] = []

    # 2. 循环records
    for record in records:
        nodes.append(
            TextNode(
                text=f"{record['label']}：{record['scope']}", # 将意图的标签和范围作为文本
                id = record["id"],
                metadata = {
                    "intent_id": record["id"],
                    "intent_label": record["label"],
                }
            )
        )

    # 3. 返回节点列表
    return nodes

# 2. 构建向量索引
def build_index(nodes :list[TextNode]) -> VectorStoreIndex:
    embed_model = util.get_embed_model()
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

# 3. 执行意图识别的检索
def retrieve(query:str, index: VectorStoreIndex, top_k: int = 1) -> list[str]:
    # 1. 执行索引的检索
    hits = index.as_retriever(similarity_top_k = top_k).retrieve(query)
    labels = []
    for hit in hits:
        labels.append(f"[命中意图]{hit.metadata.get('intent_label','')}")
    return labels

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "intent_catalog.json")
    # 2. json数组转换成node列表
    nodes = build_intent_nodes(records)
    # 3. 建立向量索引
    index = build_index(nodes)

    while True:
        query = input("\n请输入查询: ").strip()
        label = retrieve(query, index)
        print(f"识别意图：{query} （{label}）")