# 检索后处理  --->  结果过滤方案
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter,FilterOperator
from Plus import util, config

# 嵌入模型
embed_model = util.get_embed_model()

# 数据加载转换
shop_data = util.load_records(config.RETRIEVAL_DATA_PATH / "result.json")

# 构建document列表
documents = []
for item in shop_data:
    text = f"店铺：{item['shop_name']}， 介绍：{item['desc_text']}"
    metadata = {
        "doc_id": item["doc_id"],
        "distance_m": item["meta_info"]["distance_m"],
        "avg_price": item["meta_info"]["avg_price"],
        "tag_list": ",".join(item["meta_info"]["tag"]),
        "desc_text": item["desc_text"],
        "shop_name": item['shop_name'],
    }
    document = Document(text=text, metadata=metadata)
    documents.append(document)


# 构建向量索引
index = VectorStoreIndex.from_documents(documents, embed_model = embed_model)
retriever = VectorIndexRetriever(index = index, similarity_top_k=len(shop_data))

# 过滤的准则
filter = "300米之内、人均消费不超过20元、减脂可喝的三分热的奶茶"

# 从json到list[NodeWithScore]转换
raw_nodes : list[NodeWithScore] = retriever.retrieve("我想喝奶茶")
print(f"初始检索结果，共{len(raw_nodes)}条")
for n in raw_nodes:
    print(f"{n.metadata['doc_id']} | 分数:{n.score:.2f} |{n.metadata['desc_text']}")
print("-" * 60)


# 1. 相似度过滤
print("【第一层】，实现相似度过滤")
# 设置相似度阈值：低于 0.5 的检索结果直接过滤掉
# 对初始检索结果进行相似度过滤
nodes_after_sim = SimilarityPostprocessor(similarity_cutoff=0.5).postprocess_nodes(raw_nodes)
print(f"过滤后剩余 {len(nodes_after_sim)}条")
for n in nodes_after_sim:
    print(f"{n.metadata['doc_id']} | 分数:{n.score:.2f}")
print("-" * 60)


# 2. 文本规则过滤
print("【第二层】，实现文本规则过滤")
# 黑名单关键词：文本中出现这些词就直接过滤
black_words = {
    "速溶",
    "乱码",
    "违规宣传",
    "奶油雪顶",
    "炼乳奶油"
}
nodes_after_text: list[NodeWithScore] = []
for node in nodes_after_sim:
    # 获取当前节点的文本描述
    desc = node.metadata["desc_text"]
    # 空文本直接过滤
    if not desc:
        continue
    # 判断文本是否命中任意一个黑名单关键词
    hit = any(word in desc for word in black_words)
    # 没有命中黑名单才保留
    if not hit:
        nodes_after_text.append(node)
print(f"过滤后剩余 {len(nodes_after_text)}条")
for n in nodes_after_text:
    print(f"{n.metadata['doc_id']} | 分数:{n.score:.2f} | {n.metadata['desc_text']}")
print("-" * 60)


# 3. 元数据过滤
print("第三层：元数据过滤")

# 定义元数据过滤规则
metadata_filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="distance_m",
            operator=FilterOperator.LTE,
            value=300
        ),
        MetadataFilter(
            key="avg_price",
            operator=FilterOperator.LTE,
            value=20
        )
    ]
)

nodes_after_meta: list[NodeWithScore] = []

for node in nodes_after_text:
    # 判断当前节点是否满足所有 MetadataFilters 条件
    valid = True

    for filter_item in metadata_filters.filters:
        value = node.metadata.get(filter_item.key)

        # <= 条件
        if filter_item.operator == FilterOperator.LTE:
            if value is None or value > filter_item.value:
                valid = False
                break

    # 所有元数据条件都满足才保留
    if valid:
        nodes_after_meta.append(node)

print(f"过滤后剩余 {len(nodes_after_meta)}条")

for n in nodes_after_meta:
    print(
        f"{n.metadata['doc_id']} | "
        f"距离:{n.metadata['distance_m']} | "
        f"均价:{n.metadata['avg_price']}"
    )

print("-" * 60)


# 4. 轻量语义过滤
print("第四层：语义过滤")
valid_nodes = []

# 构造一个“理想答案”的标准语义描述，后面拿每个店铺描述与它做语义相似度比较，将标准描述转换成向量
ref_embedding = embed_model.get_text_embedding("低脂、无奶油、低热量、适合减脂的人群饮用")

for node in nodes_after_meta:
    # 获取店铺描述
    shop_desc = node.metadata["desc_text"]

    # 将店铺描述转换成向量
    shop_embedding = embed_model.get_text_embedding(shop_desc)

    # 计算店铺描述与标准描述之间的语义相似度
    sim_score = embed_model.similarity(
        ref_embedding,
        shop_embedding
    )

    # 相似度达到阈值，说明语义上比较符合“低脂、减脂”需求
    if sim_score >= 0.55:
        valid_nodes.append(node)

for n in valid_nodes:
    print(f"{n.metadata['doc_id']} | 店铺名称:{n.metadata['shop_name']} | 店铺的描述信息:{n.metadata['desc_text']}")
print("-" * 60)