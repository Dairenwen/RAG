# 实现后退提示方案
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter,FilterOperator, ExactMatchFilter
from Plus import util, config

# 生成后退问题的提示词模版
STEP_BACK_PROMPT = PromptTemplate(
    "你是一个超市的商品导购员。请呢将顾客的具体问题后退一步，改写成一条比较宽泛的中文导购检索描述，"
    "用来查找同类商品或者相关的商品。\n"
    "你只能输出一句后退检索描述，不要引导、不要解释。\n"
    "顾客的具体问题：{query}\n"
)

# 从用户提问中获取到元数据的标签
FILTER_FIELDS = {
    "release_year":("生产日期", True),
    "color": ("颜色", True),
    "pattern":("图案", True),
    "sub_category": ("子分类", True), # 不要把子分类作为宽泛条件
    "category":("分类", False),
    "brand":("品牌", True), # True = “你必须明确说出这个 metadata 值，我才敢拿它做硬过滤”；False = “你没明确说出来也没关系，我可以通过 embedding 猜一个”
}


# 从后退提示生成的问题中去提取元数据标签
STEP_BACK_FIELDS = {"category"} # 后退查询只允许使用“分类这些宽泛条件，不允许年份、颜色、图案、品牌限制它

# 1. 把json数组转换成节点列表
def build_nodes(records : list[dict]) -> list[TextNode]:
    nodes : list[TextNode] = []
    for record in records:
        feats = "、".join(record["features"])
        text = (
            f"{record['name']} 品牌{record['brand']} 类目{record['category']}/{record['sub_category']}\n"
            f"颜色{record['color']} 图案{record['pattern']} 年份{record['release_year']} 特性{feats}\n"
            f"{record['description']}"
        ) # 商品的关键信息都进入向量，让用户无论从“名称、类别、颜色、特性、描述”等角度提问，都有机会被语义检索命中

        nodes.append(
            TextNode(
                text=text,
                id=record["sku"],
                metadata={**record, "features": feats}
            )
        )

    return nodes


# 2. 构建向量索引
def build_index(nodes : list[TextNode]) -> VectorStoreIndex:
    return VectorStoreIndex(nodes=nodes, embed_model=util.get_embed_model())

# 3. 访问LLM
def llm_text(prompt :str) ->str:
    parts: list[str] = []
    for res in util.llm.stream_complete(prompt):
        if res.delta:
            parts.append(res.delta)
    return "".join(parts)

# 4. 把原始问题转换成后退提示之后的新问题
def step_back_query(query : str) ->str:
    raw = llm_text(STEP_BACK_PROMPT.format(query=query))
    return next((line.strip() for line in raw.splitlines() if line.strip()),"")

# 5. 从全部记录中，汇总每个可以过滤的字段和他的取值可能
def build_catalog(records : list[dict]) ->dict[str, list]:
    catalog = {k : sorted({r[k] for r in records}) for k in FILTER_FIELDS}
    catalog["features"] = sorted({f for r in records for f in r["features"]})
    return catalog

# 6. 进行key和value的拼装
def cand_text(field: str, value) -> str:
    label = FILTER_FIELDS.get(field, ("特性", False))[0]
    return f"{label}:{value}"


# 7. 从用户问题里，判断用户提到了哪些商品属性，然后生成 metadata 过滤条件
def extract_filters(
        query : str,
        catalog: dict,
        threshold: float = 0.6, # 默认阈值为0.6，表示语义相似度大于0.6的候选值才会被认为是匹配的
        active_fields: set[str] | None=None, # 这一次只允许检查哪些字段
) -> dict:
    # 1. 指定参与构造过滤条件的属性
    fields = active_fields or set(FILTER_FIELDS) | {"features"}
    # 2. 原始问题向量化处理
    q_vec = util.get_embed_model().get_text_embedding(query)
    # 3. 初始化filters
    filters = {}

    for field, (tmp, literal_only) in FILTER_FIELDS.items():
        if field not in fields:
            continue
        best, best_score = None, threshold

        for value, vec in zip(
                catalog[field],
                util.get_embed_model().get_text_embedding_batch([cand_text(field, v) for v in catalog[field]]),
                ):
            if str(value) in query:
                best = value # 先进性字面匹配，如果字面匹配到了，就不再进行语义相似度匹配
                break
            if literal_only: # 如果是只能字面匹配的字段，就不再进行语义相似度匹配
                continue
            score = util.get_embed_model().similarity(q_vec, vec) # 进行语义相似度匹配
            if score > best_score:
                best, best_score = value, score
        if best is not None:
            filters[field] = best

    # 这里选出多个特性，不再进行语义相似度匹配，只要字面匹配就行
    if "features" in fields:
        match = [f for f in catalog["features"] if f in query]

        if match:
            filters["features"] = match

    return filters

# 8 需要把filters转换成MetadataFilters对象
def build_metadata_filters(filters :dict) -> MetadataFilters:
    flist = []
    for key, value in filters.items():
        if key == "features": # 特性是一个列表，需要把每个特性都加入到过滤条件中
            for feat in value:
                flist.append(
                    MetadataFilter(
                        key="features",
                        value=feat,
                        operator=FilterOperator.CONTAINS # 特性是一个列表，需要使用CONTAINS操作符
                    )
                )
        else:
            flist.append(ExactMatchFilter(key=key, value=value))

    return MetadataFilters(filters=flist)


# 9 封装向量检索方法
def vector_search(
        query :str,
        index : VectorStoreIndex,
        filters:dict,
        top_k:int,
) -> list[NodeWithScore]:
    return index.as_retriever(similarity_top_k=top_k,filters=build_metadata_filters(filters)).retrieve(query)

# 10 封装后退提示之后的向量检索方法
def step_back_search(
        query: str,
        index : VectorStoreIndex,
        catalog:dict,
        top_k:int,
) -> tuple[list[NodeWithScore], dict]:

    filters = extract_filters(
        query,
        catalog,
        threshold=0.6,
        active_fields=STEP_BACK_FIELDS,
    )

    hits = vector_search(query, index, filters, top_k)
    return hits, filters


# 11 召回结果合并
def merge_results(orig_hits: list[NodeWithScore], back_hits:list[NodeWithScore], top_k:int) -> list[dict]:
    merged : dict[str, dict] = {}

    for hit in orig_hits:
        sku = hit.node.metadata["sku"] # 用sku作为唯一标识，避免同一个商品被重复召回
        merged[sku] = {**hit.node.metadata, "score":hit.score, "source": ["原路"]}

    for hit in back_hits:
        sku = hit.node.metadata["sku"]
        if sku in merged:
            merged[sku]["source"].append("后退")
            merged[sku]["score"] = merged[sku]["score"] = max(merged[sku]["score"], hit.score) # 如果同一个商品被原路和后退召回了，取相似度最高的
        else:
            merged[sku] = {**hit.node.metadata, "score": hit.score, "source":["后退"]}

    return sorted(merged.values(), key=lambda x: -x["score"])[:top_k]


# 格式化输出
def fmt_hit(i : int, hit:NodeWithScore, tag: str="") ->str:
    m = hit.node.metadata
    src = f" [{tag}]"
    return (
        f" {i}. {m['name']} ({m['sku']})"
        f" [{m['color']}/{m['pattern']}/{m['release_year']}] [{m['features']}]"
        f"{src} [相似度:{hit.score:.4f}]"
    )

def fmt_filters(filters: dict) -> str:
    parts = [f"{k} = {','.join(map(str, v)) if isinstance(v, list) else v}"for k, v in filters.items()]
    return "、".join(parts)

def fmt_merged(i: int, m: dict) -> str:
    return (
        f" {i}. {m['name']} ({m['sku']})"
        f" [{m['color']}/{m['pattern']}/{m['release_year']}] [{m['features']}]"
        f"[{'/'.join(m['source'])}] [相似度:{m['score']:.4f}]"
    )

def compare_step_back(
        query : str,
        index :VectorStoreIndex,
        catalog: dict,
        top_k: int=3,
) ->str:
     orig_filters = extract_filters(query, catalog)

     # 后退提示之后生成的问题
     back_q = step_back_query(query)
     orig = vector_search(query, index, orig_filters, top_k)

     if not orig:
         orig = vector_search(query, index, {}, top_k) # 如果原始问题没有召回结果，就不使用过滤条件，直接召回

     back, back_filters = step_back_search(back_q, index, catalog, top_k)

     merged = merge_results(orig, back, top_k)

     lines = [
         "【后退提示】 原始问题+后退提示+合并（召回更多的内容）",
         f"原始问题：{query}",
         f"原路过滤条件 （{len(orig_filters)}项， 严格） ：{fmt_filters(orig_filters)}",
         f"后退问句：{back_q}",
         f"后退条件（{len(back_filters)}项， 宽松）：{fmt_filters(back_filters)}",
     ]

     lines += [fmt_hit(i, h, "原路") for i, h in enumerate(orig, 1)]
     lines.append(f"------------------------- 后退宽松查询 ----------------------------")
     lines += [fmt_hit(i, h, "后退") for i, h in enumerate(back, 1)]
     lines.append(f"-------------------------- 合并去重 ------------------------------")
     lines += [fmt_merged(i, m) for i, m in enumerate(merged, 1)]
     return "\n".join(lines)


if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.PRERETRIEVE_DATA_DIR / "step_back_catalog.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 建立向量索引
    index = build_index(nodes)
    catalog = build_catalog(records)
    # 这里用过滤条件来进行严格的过滤，后退提示之后的查询会用更宽松的过滤条件，以此来突出后退提示的作用
    while True:
        query = input("请输入查询: ").strip()
        print(compare_step_back(query, index, catalog))