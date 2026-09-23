from Plus import util,config
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.schema import TextNode, NodeWithScore


# LLM的提示词模版
SUBQUERY_PROMPT = PromptTemplate(
    "你是RAG检索规划助手。用户问题往往是复合型的、多步骤的，请按照大事化小的原则进行拆分，拆分成若干个子问题，"
    "每条子问题都可以去向量索引中去进行单独的检索，合并在一起又能覆盖原始的查询问题。\n"
    "要求:\n"
    "1. 子问题必须要紧扣原意，不要引入新的未涉及的话题\n"
    "2. 假如原始的问题足够简单，输出原始的问题即可\n"
    "3. 将来输出的子问题，每一行都不要编号。最多{max_subqueries} 条\n\n"
    "用户的原始问题：{query}\n"
)

# 1. 把json数组转换成TextNode列表
def build_nodes(records : list[dict]) -> list[TextNode]:
    # 1. 创建空列表
    nodes : list[TextNode] = []
    # 2. 遍历records
    for record in records:
        nodes.append(
            TextNode(
                text=record["text"],
                id = record["id"],
                metadata = {
                    "id": record["id"],
                    "title": record["title"],
                    "text": record["text"],
                }
            )
        )

    return nodes

# 2. 构建向量索引
def build_index(nodes :list[TextNode]) -> VectorStoreIndex:
    embed_model = util.get_embed_model()
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

# 3. 调用LLM流式生成查询文本
def generate_text(query:str, num_queries: int) ->str:
    # 1. 构建提示词
    prompt = SUBQUERY_PROMPT.format(query = query.strip(),max_subqueries=num_queries)

    # 2. 最终的结果
    full_text = ""

    for res in util.llm.stream_complete(prompt):
        full_text += res.delta or ""

    return full_text




# 4. 子查询扩展
def decompose_subqueries(query:str, num_queries:int)->list[str]:
    # 1. 调用LLM生成多条子查询
    multi_query_text = generate_text(query,num_queries)
    return multi_query_text.split("\n")

# 5. 执行检索操作
def retrieve_hits(query: str, index: VectorStoreIndex, top_k:int) ->list[NodeWithScore]:
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)

# 6. 合并多路查询检索结果的逻辑
def merge_hits(hits_list: list[list[NodeWithScore]]) -> list[NodeWithScore]:
    bestret : dict[str, NodeWithScore] = {}

    for hits in hits_list:
        for hit in hits:
            node_id = hit.node.node_id
            if node_id not in bestret or hit.score > bestret[node_id].score:
                bestret[node_id] = hit # 保留相似度最高的命中结果

    return sorted(bestret.values(), key=lambda x : x.score, reverse=True)


# 7. 格式化输出内容
def format_hit(rank: int, hit: NodeWithScore) -> str:
    title = hit.node.metadata.get("title")
    return f"{rank}. {title} [相似度:{hit.score:.4f}]"


# 8. 普通查询的流程
def compare_sub_query(
        query: str,
        index: VectorStoreIndex,
        top_k: int = 3
) -> str:
    # 1. 用原始的问题直接进行向量检索
    hits = retrieve_hits(query, index, top_k)

    # 2. 标准输出
    lines = [
        "【普通查询】 仅使用原始问题进行查询",
        f"查询：{query}",
        f"命中 {top_k}:"
    ]

    for i, hit in enumerate(hits, 1):
        lines.append(format_hit(i, hit))
    return "\n".join(lines)


# 9. 多路查询的流程
def compare_multi_query(
        query: str,
        index: VectorStoreIndex,
        num_queries: int = 3,
        top_k: int = 3
) -> str:
    # 1. 得到扩写的查询
    queries = decompose_subqueries(query, num_queries)
    # 2. 针对扩写后的查询列表，逐一检索
    hits = [retrieve_hits(query, index, top_k) for query in queries]
    # 3. 多路查询检索得到的结果进行合并
    merged = merge_hits(hits)[:top_k]

    # 4. 格式化输出
    lines = [
        "【子查询】 使用原始问题的子集变体进行查询",
        f"原始问题：{query}",
        "子集查询：",
    ]

    for query in queries:
        lines.append(f"{query}")

    lines.append(f"命中 {top_k}: ")

    if not merged:
        lines.append("（没有命中）")
    else:
        lines.extend(format_hit(i, hit) for i, hit in enumerate(merged, 1))

    return "\n".join(lines)


if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "multi_query_corpus.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 建立向量索引
    index = build_index(nodes)

    while True:
        query = input("请输入查询：").strip()
        print(compare_sub_query(query, index))
        print("--------------------------------")
        print(compare_multi_query(query, index))