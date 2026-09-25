# 稠密检索
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from Plus import util, config

# 把json数组转换成list[TextNode]
def build_nodes(records : list[dict]) ->list[TextNode]:
    nodes : list[TextNode] = []

    for record in records:
        text = f"病症：{record['disease_name']}，症状描述：{record['symptom_text']}"
        nodes.append(
            TextNode(
                text=text,
                id = record["doc_id"],
                metadata={
                    "disease_name":record['disease_name'],
                }
            )
        )

    return nodes


# 创建稠密检索的检索器
def create_retriever(
        nodes :list[TextNode],
) -> VectorStoreIndex:
    return VectorStoreIndex(
        nodes=nodes,
        embed_model=util.get_embed_model(),
    )


if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.RETRIEVAL_DATA_PATH / "retrieval.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 建立检索器
    retriever = create_retriever(nodes)
    while True:
        query = input("请输入查询: ").strip()
        hits = retriever.as_retriever(similarity_top_k=5).retrieve(query)
        for hit in hits:
            disease_name = hit.node.metadata.get("disease_name")
            score = hit.score
            print(f"【症状】：{disease_name} 【相似度】 {score:.4f}")