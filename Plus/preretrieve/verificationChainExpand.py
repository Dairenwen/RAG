# 实现验证链
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from Plus import config, util

# 生成的提示词的模版
DRAFT_PROMPT = PromptTemplate(
    "你是一个RAG检索规划助手。根据用户的问题写出最多 {max_subqueries} 条检索子查询，要求如下\n"
    "1. 子查询需要紧扣愿意，不要引入额外的新话题；\n"
    "2. 按照检索顺序排列 （先定义，再流程，最后结论）\n"
    "3. 子查询按行输出，不要编号，不要解释。\n\n"
    "原始问题：{query}\n"
    "草稿子查询：\n"
)

# 验证环节的提示词模版
VERIFY_PROMPT = PromptTemplate(
    "你是一个RAG检索审核助手。审核下列子查询是否与原始问题相关、是否重复、是否跑题。\n"
    "要求如下：\n"
    "1. 审核过程逐行写：子查询：\n"
    "2. 不要编号，不要解释, 只保留子查询项\n"
    "3. 从待审核的子查询中提取与原问题的含义最贴近的：{nums}条\n"
    "原始问题：{query}\n"
    "待审核的子查询：\n{drafts}\n"
)

# 1. 把json数组转换成list[TextNode]
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

# 3. 调用LLM流式生成文本
def generate_text(prompt : str) -> str:
    parts: list[str] = []
    for res in util.llm.stream_complete(prompt):
        if res.delta:
            parts.append(res.delta)
    return "".join(parts)

# 4. 调用LLM生成变体查询
def generate_drafts(query:str, max_subqueries: int) -> list[str]:
    prompt = DRAFT_PROMPT.format(query = query.strip(), max_subqueries =max_subqueries)
    drafts = [line.strip() for line in generate_text(prompt).split("\n") if line.strip()]
    return drafts

# 5. 调用LLM验证前面生成的变体查询
def verify_drafts(query:str, drafts:list[str], nums:int) -> list[str]:
    prompt = VERIFY_PROMPT.format(query=query.strip(), drafts="\n".join(drafts), nums=nums)
    return [line.strip() for line in generate_text(prompt).split("\n") if line.strip()][:nums]

# 6. 把生成与验证环节链式处理
def expand_verification_chain(
    query:str,
    max_subqueries:int =6,
    nums:int=3,
) ->tuple[list[str], list[str]]:
    drafts = generate_drafts(query, max_subqueries)
    final = verify_drafts(query, drafts, nums)
    if not final:
        final = drafts[:nums]
    return  drafts, final

# 7. 单条查询的检索逻辑
def retrieve_hits(query: str, index: VectorStoreIndex, top_k:int) ->list[NodeWithScore]:
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)

# 8. 合并查询结果
def merge_hits(hits_list: list[list[NodeWithScore]]) -> list[NodeWithScore]:
    best : dict[str, NodeWithScore] = {}

    for hits in hits_list:
        for hit in hits:
            node_id = hit.node.node_id
            if node_id not in best or hit.score > best[node_id].score:
                best[node_id] = hit

    return sorted(best.values(), key=lambda h : h.score, reverse=True)

# 9. 格式化输出查询到的内容
def format_hit(rank :int, hit:NodeWithScore) -> str:
    title = hit.node.metadata.get("title")
    return f" {rank}. {title} [相似度:{hit.score:.4f}]"

# 10. 普通查询
def compare_single_query(
    query: str,
    index: VectorStoreIndex,
    top_k: int =3
) ->str:
    # 1. 用原始的问题直接进行向量检索
    hits = retrieve_hits(query, index, top_k)
    # 2. 标准输出
    lines = [
        "【普通查询】 仅使用原始问题进行查询",
        f"查询：{query}",
        f"命中 {top_k}:"
    ]
    lines.extend(format_hit(i, hit) for i, hit in enumerate(hits, 1))
    return "\n".join(lines)

# 11. 验证链查询结果
def compare_verification_chain_query(
    query: str,
    index: VectorStoreIndex,
    max_subqueries: int=6,
    nums:int=3,
    top_k:int=3
) ->str:
    drafts, final = expand_verification_chain(query, max_subqueries, nums)
    hits = [retrieve_hits(sq, index, top_k) for sq in final]
    merged = merge_hits(hits)
    lines = [
        "【验证链】 草稿子查询经过审核后分别检索",
        f"原始问题：{query}",
        "草稿子查询：",
    ]
    for i, d in enumerate(drafts, 1):
        lines.append(f" {i}. {d}")

    lines.append("验证后的子查询：")
    for i, d in enumerate(final, 1):
        lines.append(f" {i}. {d}")

    lines.append(f"合并命中 {top_k}：")
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
        query = input("\n请输入查询: ").strip()
        print(compare_single_query(query, index))
        print("--------------------------------")
        print(compare_verification_chain_query(query, index))