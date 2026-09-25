# 实现元数据路由的效果：  LLM-->提取科室 --->路由科室  --> 向量检索
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from Plus import config, util


# 提示词模版
ROUTE_PROMPT = PromptTemplate(
    "你是一个医院分诊助手。请根据患者的描述，从下列科室中选择出患者应该前往的科室，只返回提供的科室名称即可。\n"
    "可供选择的科室：{departments}\n"
    "患者的描述：{query}\n"
    "要求如下：\n"
    "1. 只能从提供的科室中去选择，不要胡编乱造\n"
    "2. 返回的时候可以返回一个科室，也可以返回多个科室，多个科室中间要用中文的逗号分隔开\n"
    "3. 只需要输出科室名称，不要过多解释\n"
)

# 1. 把json数组转换成节点列表
def build_nodes(records :list[dict]) -> list[TextNode]:

    nodes : list[TextNode] = []

    for record in records:
        nodes.append(
            TextNode(
                text = f"{record['department']}:{record['description']}",
                id = record["room_id"],
                metadata={
                    "room_id": record["room_id"],
                    "room_name": record["room_name"],
                    "department": record["department"],
                    "doctor": record["doctor"],
                    "floor": record["floor"],
                    "description": record["description"],
                }
            )
        )

    return nodes

# 2. 按照科室分组构建多个向量索引
def build_indexes(records :list[dict]) -> dict[str, VectorStoreIndex]:
    embed_model = util.get_embed_model()
    grouped: dict[str, list[dict]] = {}

    for record in records:
        grouped.setdefault(record["department"], []).append(record)

    result = {}

    for dept, dept_records in grouped.items():
        result[dept] = VectorStoreIndex(
            nodes=build_nodes(dept_records),
            embed_model=embed_model
        )
    return result

# 3. 从json数组中获取所有的科室
def build_departments(records : list[dict]) -> set[str]:
    return set(record["department"] for record in records)

# 4. 调用LLM生成科室
def generate_text(prompt :str) ->str:
    parts: list[str] = []
    for res in util.llm.stream_complete(prompt):
        if res.delta:
            parts.append(res.delta)
    return "".join(parts)

# 5. llm生成路由标签
def extract_route(query :str, departments: list[str]) -> list[str]:
    dept_options = "、".join(departments)
    raw = generate_text(
        ROUTE_PROMPT.format(departments=dept_options,query=query.strip()).strip()
    )

    return raw.split("，")

# 6. 根据生成的标签，去执行向量检索
def retrieve_hits(
        query :str,
        indexes : dict[str, VectorStoreIndex],
        top_k : int=5,
        route: list[str] | None = None,
) ->list[NodeWithScore]:

    hits : list[NodeWithScore] = []

    # 2. 遍历路由逻辑
    for index in [
        indexes[dept]
        for dept in route if dept in indexes.keys()
    ]:
        hits.extend(index.as_retriever(similarity_top_k =top_k).retrieve(query))

    # 3. 排序提取
    hits.sort(key = lambda hit : hit.score, reverse=True)

    return hits[:top_k]


# 7. 格式化输出检索到的内容
def format_hit(rank:int , hit : NodeWithScore) ->str:
    metadata = hit.node.metadata
    return (
        f" {rank}. {metadata['room_name']} （{metadata['room_id']}）"
        f" [{metadata['department']}/{metadata['floor']}/医生{metadata['doctor']}]"
        f" [相似度：{hit.score:.4f}]"
    )


# 8 对外提供的路由检索调用方法
def route_query(
        query: str,
        indexes: dict[str, VectorStoreIndex],
        departments: list[str],
        top_k: int = 5,
) -> str:
    route = extract_route(query, departments)
    hits = retrieve_hits(query, indexes, top_k, route=route)
    route_text = "，".join(route)
    index_text = "，".join(route)
    lines = [
        "【元数据路由】",
        f"患者描述：{query}",
        f"可选科室：{"、".join(departments)}",
        f"路由科室：{route_text}",
        f"检索索引：{index_text}",
        f"命中{len(hits)}条："
    ]
    lines.extend(format_hit(i, hit) for i, hit in enumerate(hits, 1))
    return "\n".join(lines)

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "metadata_route.json")
    # 3. 解析出来所有的科室
    departments = build_departments(records)
    # 4. 建立向量索引
    indexes = build_indexes(records)

    while True:
        query = input("请输入查询: ").strip()
        print(route_query(query, indexes, departments))