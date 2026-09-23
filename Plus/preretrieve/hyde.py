# 实现假设文档嵌入的方案
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore

from Plus import config, util

# 提示词模版
HYDE_PROMPT = PromptTemplate(
    "你是一个大学校园服务 HYDE辅助模型。根据大学生咨询的问题，写一段 2~6句的一份政策说明文档，"
    "文档的风格要严肃、严谨；文风像权威部门比如教务处、学生处发通知的风格；生成的文档要尽可能仿真。\n"
    "格式要求：直接输出正文：不要疑问句，不要问答的形式\n"
    "根据问题涉及的具体业务，合理使用大学校园相关的规范术语，不要引入与问题无关的概念。\n"
    "学生咨询的问题：{query}\n"
)


# 1. 把json数组转换成节点列表
def build_nodes(records: list[dict]) ->list[TextNode]:

    nodes: list[TextNode] = []

    for record in records:
        text = f"{record['title']}\n{record['category']}：{record['content']}"
        nodes.append(
            TextNode(
                text=text,
                id = record["doc_id"],
                metadata = {
                    "doc_id":record["doc_id"],
                    "category":record["category"],
                    "title":record["title"],
                    "content":record["content"],
                }
            )
        )
    return nodes

# 2. 构建向量索引
def build_index(nodes :list[TextNode]) -> VectorStoreIndex:
    embed_model = util.get_embed_model()
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

# 3. 调用LLM生成假设文档
def generate_text(prompt:str) ->str:
    parts : list[str] = []
    for res in util.llm.stream_complete(prompt):
        if res.delta:
            parts.append(res.delta)
    return "".join(parts)

# 4. 生成假设文档
def generate_hyde_document(query:str) ->str:
    prompt = HYDE_PROMPT.format(query = query)
    raw = generate_text(prompt)
    text = " ".join(ln.strip() for ln in raw.splitlines())
    return text

# 5. 单条检索
def retrieve_hits(query: str, index: VectorStoreIndex, top_k:int) ->list[NodeWithScore]:
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)

# 6. 格式化输出检索到的内容
def format_hit(rank :int, hit:NodeWithScore) -> str:
    title = hit.node.metadata.get("title")
    category = hit.node.metadata.get("category")
    return f" {rank}. {title} [{category}] [相似度:{hit.score:.4f}]"

# 7. 普通检索，直接拿原始的问题去向量索引中检索
def compare_single_query(
        query: str,
        index: VectorStoreIndex,
        top_k: int =3
) ->str:
    # 1. 用原始的问题直接进行向量检索
    hits = retrieve_hits(query, index, top_k)
    # 2. 标准输出
    lines = [
        "【普通查询】:仅使用原始问题进行查询",
        f"查询：{query}",
        f"命中 {top_k}:"
    ]
    lines.extend( format_hit(i, hit) for i, hit in enumerate(hits, 1) )
    return "\n".join(lines)

# 8. 假设文档嵌入检索
def compare_hyde_query(
        query: str,
        index: VectorStoreIndex,
        top_k: int =3
) -> str:
    # 1. 先根据用户原始问题生成假设文档
    hypo = generate_hyde_document(query)
    # 2. 执行假设文档的检索
    hits = retrieve_hits(hypo, index, top_k)
    # 3. 按照标准格式输出
    lines = [
        "【HyDE 假设文档嵌入】: LLM生成原始问题的假设文档再去做检索",
        f"原始问题：{query}",
        f"假设文档：{hypo}",
        f"命中 {top_k}",
    ]
    lines.extend(format_hit(i, hit) for i, hit in enumerate(hits, 1))
    return "\n".join(lines)

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "hyde.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 建立向量索引
    index = build_index(nodes)
    while True:
        query = input("\n请输入查询: ").strip()
        print("\n" + compare_single_query(query, index))
        print("--------------------------------")
        print("\n" + compare_hyde_query(query, index))