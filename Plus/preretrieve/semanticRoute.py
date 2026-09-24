# 实现语义路由：
# 嵌入模型预测品类 → 路由到对应品类索引 → 向量检索
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from Plus import config, util

ROUTE_DESC = {
    "儿童玩具": "儿童玩耍、过家家、仿真道具、玩具",
    "厨房厨具": "家用烹饪、炒菜、炖锅、餐具",
    "运动器材": "户外健身、打球、跳绳、运动装备"
}

# 1. 把json数组转换成节点列表
def build_nodes(records : list[dict]) -> list[TextNode]:

    nodes : list[TextNode] = []

    for record in records:
        nodes.append(
            TextNode(
                text= f"物品名称：{record['item_name']}，描述：{record['item_desc']}",
                id = record["item_name"],
                metadata={
                    "item_name": record["item_name"],
                    "item_desc": record["item_desc"],
                    "meta_label": record["meta_label"], # 这个属性大多数是错误的
                    "predicted_category":record["predicted_category"], # 预测的品类，这个是我们通过嵌入模型预测出来的，可能是正确的，也可能是错误的
                    "real_category":record["real_category"], # 真实的品类，这里只是为了验证模型的预测结果是否正确，真实的品类不应该作为检索的依据
                }

            )
        )

    return nodes

# 2. 借助嵌入模型，对缺失的或者错误的分类标签进行修正
def predict_categories(
        text: str,
        categories: list[str],
        threshold: float = 0.3
) -> list[str]:
    # 1. 获取嵌入模型
    embed_model = util.get_embed_model()

    # 2. 将商品描述转换为向量
    text_emb = embed_model.get_text_embedding(text)

    # 3. 保存所有满足相似度阈值的分类
    result = []

    # 4. 遍历所有候选分类
    for cat in categories:
        # 将当前分类的描述转换为向量
        desc_emb = embed_model.get_text_embedding(ROUTE_DESC[cat])

        # 计算商品描述与分类描述之间的语义相似度
        sim = embed_model.similarity(text_emb, desc_emb)

        # 5. 相似度达到阈值，则认为属于该分类
        if sim >= threshold:
            result.append(cat)

    # 6. 返回所有符合条件的分类
    return result

# 2. 按照分组构建多个向量索引
def build_indexes(
        records: list[dict],
        categories: list[str]
) -> dict[str, VectorStoreIndex]:

    embed_model = util.get_embed_model()

    # 按分类保存商品
    grouped: dict[str, list[dict]] = {}

    # 1. 遍历每个商品
    for record in records:

        # 得到该商品可能属于的多个分类
        predicted_categories = predict_categories(record["item_desc"],categories)

        # 2. 一个商品可能加入多个分类
        for cat in predicted_categories:
            grouped.setdefault(cat, []).append({**record,"predicted_category": cat})

    # 3. 为每个分类分别构建向量索引
    result = {}

    for category, category_records in grouped.items():
        result[category] = VectorStoreIndex(
            nodes=build_nodes(category_records),
            embed_model=embed_model
        )

    return result

# 3. 获取所有的真实的品类
def build_categories() -> list[str]:
    return list(ROUTE_DESC.keys())

# 6. 根据生成的标签，去执行向量检索
def retrieve_hits(
        query: str,
        indexes: dict[str, VectorStoreIndex],
        top_k: int = 5,
        route: list[str] | None = None,
) -> list[NodeWithScore]:

    hits: list[NodeWithScore] = []

    # 1. 如果没有指定路由，则默认搜索所有分类
    if route is None:
        route = list(indexes.keys())

    # 2. 遍历命中的多个分类索引进行检索
    for dept in route:
        if dept in indexes:
            retriever = indexes[dept].as_retriever(similarity_top_k=top_k)
            hits.extend(retriever.retrieve(query))

    # 3. 按相似度从高到低排序
    hits.sort(
        key=lambda hit: hit.score or 0,
        reverse=True
    )

    # 4. 去重，避免同一个商品在多个分类中重复出现
    result = []
    seen = set()

    for hit in hits:
        text = hit.node.get_content()

        if text not in seen:
            seen.add(text)
            result.append(hit)

        if len(result) >= top_k:
            break

    return result


# 7. 格式化输出检索到的内容
def format_hit(rank:int, hit: NodeWithScore) -> str:
    metadata = hit.node.metadata
    meta_label = metadata["meta_label"] or "（空）"
    return (
        f" {rank}. {metadata['item_name']}"
        f" [推断品类：{metadata['predicted_category']} / 真实品类：{metadata['real_category']}"
        f" / 实际的标签：{meta_label}]"
        f" [相似度：{hit.score:.4f}]"
    )



# 8. 将来对外提供的语义路由检索方法
def route_query(
        query: str,
        indexes: dict[str, VectorStoreIndex],
        categories: list[str],
) -> str:
    route = predict_categories(query, categories)
    hits = retrieve_hits(query, indexes, route=route)
    lines = [
        "【语义路由】",
        f"用户查询：{query}",
        f"可选品类：{"、".join(categories)}",
        f"路由品类：{route}",
    ]
    lines.extend( format_hit(i, hit) for i, hit in enumerate(hits, 1)  )
    return "\n".join(lines)


if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "semantic_route.json")
    categories = build_categories()
    # 建立向量索引
    indexes = build_indexes(records, categories)
    while True:
        query = input("请输入查询: ").strip()
        print(route_query(query, indexes, categories))