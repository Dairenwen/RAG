from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core.graph_stores import EntityNode
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.schema import TransformComponent, BaseNode, Sequence, Any, MetadataMode
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import TextNode
from Plus import util,config
from typing import Sequence, Any
from collections import deque

# 1.将数据注入三元组
def make_kg_extractor(domain_by_id : dict[str, dict]) -> TransformComponent:

    # 内部类
    class KgExtractor(TransformComponent):

        @classmethod # 起名字
        def class_name(cls) -> str:
            return  "KgExtractor"

        # 让内部类当作函数被调用
        def __call__(
                self,
                nodes: Sequence[BaseNode],
                show_progress: bool = False, # 不显示进度条
                **kwargs: Any,
        ) -> Sequence[BaseNode]:
            # 逐个处理节点
            for node in nodes:
                domain = domain_by_id[node.metadata["domain_id"]]
                kg_nodes: list[EntityNode] = []
                kg_relations: list[Relation] = []

                meta = {"domain_id": domain["id"]}

                for relation in domain["relations"]:
                    source_node = EntityNode(
                        name=relation["source"],
                        label="ENTITY",
                        properties=meta,
                    )

                    target_node = EntityNode(
                        name=relation["target"],
                        label="ENTITY",
                        properties=meta,
                    )
                    kg_nodes += [source_node, target_node]

                    kg_relations.append(
                        Relation(
                            label=relation["label"],
                            source_id=source_node.id,
                            target_id=target_node.id,
                            properties=meta,
                        )
                    )

                node.metadata["nodes"] = kg_nodes
                node.metadata["relations"] = kg_relations

            return nodes

    return KgExtractor()



# 2. 把JSON数组转换成list[TextNode]
def build_nodes(domains: list[dict]) -> list[TextNode]:
    nodes: list[TextNode] = []

    for domain in domains:
        nodes.append(
            TextNode(
                text=domain["scope"],
                id_=f"{domain['id']}",
                metadata={"domain_id": domain["id"]},
            )
        )
    return nodes

# 3. 构建图索引
def build_index(domains: list[dict]) -> PropertyGraphIndex:
    # 1. 转换节点列表
    nodes = build_nodes(domains)
    # 2. 创建图属性的索引
    graph_index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[ make_kg_extractor({domain["id"] : domain for domain in domains }) ],
        # 提取器列表，这些提取器会作用到节点上，用来提取图中的 triplets / 实体关系
        embed_model = util.get_embed_model(),
    )
    # 3. 返回图索引的对象
    return graph_index


# 4. 判断当前用户查询的问题归属与哪个知识域 (这里图比较少，利用LLM来判断),返回关于子图的所在domain
def pick_domain(query: str, domains: list[dict]) -> dict | None:
    # 1. 整理每个域发送给大模型的内容
    domain_lines = "\n".join(
        f"* id={domain['id']}，title={domain['title']}，scope={domain['scope']}"
        for domain in domains
    )
    # 2. 统计一下所有id的集合
    valid_ids = {domain["id"] for domain in domains}
    # 3. 提示词
    prompt = (
        "你是一个知识域的分类器。根据用户的问题，从下列知识域中选择出来最匹配的一个。\n"
        "只允许输出该域的id，不要输出任何无关的信息。\n"
        "假如用户的问题与所有的知识域无关，直接输出 none。\n"
        f"可供选择的知识域：\n{domain_lines}\n\n"
        f"用户的问题：{query}\n"
        "你只能输出id或者none"
    )
    # 4. 访问LLM
    response = util.llm.complete(prompt=prompt)
    # 5. 分析返回的结果
    text = response.text.strip().lower()

    if text in {"none", "null", "无效", "无关"}: # 增加容错，防止大模型输出了其他的无效词
        return None

    # 6. 解析domain_id，防止出现反模型返回 text = "我认为应该是 history"
    domain_id = text if text in valid_ids else next(
        (domain["id"] for domain in domains if domain["id"] in text),
        None
    )
    # 7. 判断获取到的domain_id
    if domain_id is None:
        return None

    return next((domain for domain in domains if domain["id"] == domain_id),None)


# 5. 从用户的查询问题中去解析出来实体名称
def resolve_entities(query:str, domain: dict) -> list[str]:
    names : list[str] = []

    surfaces = sorted(
        [(e["name"], e["name"]) for e in domain["entities"]], # 这里可以扩展为别名->规范名的映射
        key = lambda  item : len(item[0]),
        reverse=True
    )

    for surface, canonical in surfaces:
        if surface in query and canonical not in names: # 判断规范名是否存在
            names.append(canonical)

    for relation in domain["relations"]:
        pattern = f"{relation['source']}的{relation['label']}"
        if pattern in query and relation["target"] not in names:
            names.append(relation["target"])
        # target 在 BFS 前其实疑似已经确定，后面的bfs更像是在“查找/展示这两个实体之间的路径”

    return names


# 6. 给原始 query 补充相似实体名，方便后面的图向量检索
#（当第 5 步没有办法形成有效 BFS 结果时，再用向量检索找和 query 语义相近的实体，
# 把这些实体名补到 query 里，帮助后面的 PropertyGraphIndex 检索）
def build_graph_search_query(query:str, domain:dict, top_k: int=5 )-> str:
    # VectorStoreIndex
    # 1. 节点列表
    nodes = [
        TextNode(
            text = entity["name"],
            metadata={"entity_name": entity["name"]},
        )
        for entity in domain["entities"]
    ]

    hits = VectorStoreIndex(nodes=nodes, embed_model=util.get_embed_model()).as_retriever(
        similarity_top_k = min(top_k, len(nodes))
    ).retrieve(query)

    # 2. 定义存放相似实体名的数组
    similar_entities : list[str] = []
    for hit in hits:
        name = hit.node.metadata["entity_name"]
        if name not in similar_entities:
            similar_entities.append(name)

    if not similar_entities:
        return query

    # 3. 把原始的query与相似的实体名拼在一起
    return f"{query} {' '.join(similar_entities)}"


# 7. 采用BFS去遍历图中的边，找到两个节点之间的关系
def find_paths(domain:dict, source:str, target:str, limit: int =5) ->list[list[tuple[str, str]]]:
    # 1. 邻接表
    adj : dict[str, list[tuple[str, str]]] = {}

    # 2. 遍历所有的边（都按照有向边）
    for relation in domain["relations"]:
        adj.setdefault(relation["source"], []).append((relation["target"],relation["label"]))

    # 3. 需要BFS队列
    queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(source,[(source, "")])]) # 初始化时第一个("刘备", "")

    # 4. 记录返回的路径
    paths : list[list[tuple[str, str]]] = []

    # 5. BFS主循环
    while queue and len(paths) < limit:
        node, path = queue.popleft()
        # 节省计算量
        if len(path) > 10: # 如果路径太长直接跳过
            continue

        for neighbor, label in adj.get(node, []):
            if any(neighbor == step[0] for step in path):
                continue

            new_path = path + [(neighbor, label)]

            if neighbor == target:
                paths.append(new_path)
            elif len(new_path) <= 10: # 如果路径太长就不继续BFS了
                queue.append((neighbor, new_path))

    return paths

# 8. 把查到的关联关系进行格式化
def format_path(path: list[tuple[str, str]]) -> str:
    text = path[0][0]

    for name, label in path[1:]:
        text += f"-[{label}] -> {name}"
    return text

# 9. 按照图中遍历的方式回答用户的提问
def graph_answer(query: str, domain: dict) -> str:
    # 1. 从query中解析实体
    entities = resolve_entities(query, domain)

    # 2. 判断entities
    if not entities:
        return ""
    # 从用户问题里找出相关实体，如果没有找到内容，后面会走图向量检索兜底

    # 3. 统计路径
    lines: list[str] = []

    for i , entity_a in enumerate(entities):
        for entity_b in entities[i+1:]:
            paths = find_paths(domain, entity_a, entity_b)
            path_text = ": ".join(format_path(path) for path in paths) if paths else "无路径"
            lines.append(f"[{entity_a}] -> [{entity_b}] : {path_text}")

    return "\n".join(lines)

# 10. 图向量索引检索
def retrieve_graph_hits(
        query:str,
        index:PropertyGraphIndex,
        domain:dict,
        top_k: int=3
) -> list[str]:
    results: list[str] = []
    # 1. 先来用嵌入模型进行实体的扩写
    search_query = build_graph_search_query(query, domain)

    # 2. 调用图索引进行查询
    hits = index.as_retriever(
        sub_retrievers=[
            VectorContextRetriever( # VectorContextRetriever 返回的 hit.node 具体内容格式主要由 LlamaIndex 内部构造决定
                graph_store=index.property_graph_store,
                vector_store=index.vector_store,
                embed_model=util.get_embed_model(),
                similarity_top_k=top_k,
            )
        ]
    ).retrieve(search_query)

    # 3. 遍历查询结果，提取文本
    for hit in hits:
        content = (hit.node.get_content(metadata_mode=MetadataMode.NONE) or "")
        if content and content not in results:
            results.append(content)

    return results


# 11. 封装对外提供的检索方法
def retrieve(
        query:str,
        domains: list[dict],
        index: PropertyGraphIndex,
        top_k: int =3
) -> str:

    # 1. LLM选域
    domain = pick_domain(query, domains)

    # 2. 判断选域结果
    if domain is None:
        return "没有命中任何与，检索内容与知识库内容无关"

    # 3. BFS遍历查询结果
    text = graph_answer(query, domain)

    lines = [
        "【场景】 知识图谱检索",
        f"用户输入内容：{query}",
        f"内容命中的知识域：{domain['title']}",
    ]

    if text: # 如果BFS有结果，就直接返回BFS的结果
        lines.extend([
            " --- 知识图谱检索 --- ",
            f" {text} ",
        ])
    else: # 如果BFS没有结果，就走图向量索引检索兜底
        graph_text = retrieve_graph_hits(query, index, domain, top_k = top_k)
        if graph_text:
            lines.append(" --- 图向量索引检索 --- ")
            for str in graph_text:
                lines.append(f"{str}：")

    return "\n".join(lines)

# * BFS 的优点是**精确、可解释**：如果能识别出明确实体，就直接查实体之间的图路径，不需要做模糊语义检索。
# * 但 BFS 很依赖前面的 `resolve_entities()` 能不能准确识别实体。比如用户说“玄德”和数据里只有“刘备”，或者用户表达得比较模糊，就可能识别失败。
# * 这时 `retrieve_graph_hits()` 会先用向量检索找与 query 语义相近的实体，把这些实体补到 query 中，再交给 `PropertyGraphIndex` 做图向量检索。
# * 所以它作为兜底的原因就是：**BFS 适合“实体明确、关系明确”的精确查询；图向量检索适合“实体没说准、表达模糊”的语义查询。**

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.CHUNK_DATA_DIR / "knowledgeGraphChunk.json")
    # 2. 建立图索引
    index = build_index(records)
    while True:
        try:
            query = input("\n请输入查询: ").strip()
        except Exception:
            print("\n退出")
            break

        if query.lower() in ("q"):
            print("\n退出")
            break

        print("\n" + retrieve(query, records, index))