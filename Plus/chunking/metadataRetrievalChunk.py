from llama_index.core.vector_stores import MetadataFilter, ExactMatchFilter, MetadataFilters

from Plus import util,config
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import VectorStoreIndex


# 统计一下哪些属性可以参与分类
FIELD_LABLES={
    "title": "标题",
    "author": "作者",
    "category": "类别",
    "floor": "楼层",
    "zone": "区域",
    "status": "状态",
}

# 统计一下哪些属性可以直接跳过
SKIP_LABLES={
    "book_id",
    "text",
}

# 1.汇总树木中metadata的所有取值
def build_vocab(records:list[dict])->dict[str,set[str]]:
    vocab:dict[str,set[str]]={}
    for record in records:
        for key,value in record.items():
            if key in SKIP_LABLES:
                continue
            vocab.setdefault(key, set()).add(value)
    return vocab

# 2.从用户输入的问题中，去解析出过滤条件，当同一个过滤字段识别出了多个候选值时，只保留最长、通常也是最具体的那个值。
def filters_from_query(
    query: str,
    vocab: dict[str, set[str]]
) -> dict[str, str]:

    filtered_records: dict[str, str] = {}

    pairs = [
        (key, value)
        for key, values in vocab.items()
        for value in values
    ]

    pairs.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )
    # 注意这里采用逆序遍历，保证当同一个过滤字段识别出了多个候选值时，只保留最长、通常也是最具体的那个值。
    # 遍历用户的query来产出metadata的过滤条件
    for key, value in pairs:
        if value in query and key not in filtered_records:
            filtered_records[key] = value

    return filtered_records

# 4.把json数组转换成list[TextNode]
def build_metadata_nodes(records:list[dict])->list[TextNode]:
    nodes:list[TextNode]=[]
    for record in records:
        metadata={key:value for key,value in record.items() }
        embedding_text = (
            f"书名：{record['title']}\n"
            f"作者：{record['author']}\n"
            f"类别：{record['category']}\n"
            f"内容：{record['text']}"
        )
        nodes.append(TextNode(
            text=embedding_text, # 进行embedding时，优化为详细的文本内容，便于后续进行语义检索，命中率更高
            id=f"{record['book_id']}::metadata",
            metadata=metadata # metadata中包含了所有的属性，便于后续进行过滤
        ))
    return nodes

# 5.构建索引
def build_index(nodes:list[TextNode])->VectorStoreIndex:
    embed_model=util.get_embed_model()
    return VectorStoreIndex(nodes=nodes,embed_model=embed_model)

# 6.可视化最后的命中结果，打印全部属性
def visualize_hits(hits:list[NodeWithScore])->list[str]:
    results=[]
    for i,hit in enumerate(hits):
        metadata=hit.node.metadata
        result=f"【命中书籍属性】{i+1}.\n"
        result+=f"[命中序号]{metadata.get('book_id','')}\n"
        result+=f"[命中书名]{metadata.get('title','')}\n"
        result+=f"[命中作者]{metadata.get('author','')}\n"
        result+=f"[命中类别]{metadata.get('category','')}\n"
        result+=f"[命中楼层]{metadata.get('floor','')}\n"
        result+=f"[命中区域]{metadata.get('zone','')}\n"
        result+=f"[命中状态]{metadata.get('status','')}\n"
        result+=f"[命中描述]{metadata.get('text','')}\n"
        result+=f"相似度:{hit.score:.5f}\n"
        results.append(result)
    return results

# 7.把过滤条件进行格式化
def format_filters(filters:dict[str,str])->str:
    lines=[]
    for key,value in filters.items():
        lines.append(f"{FIELD_LABLES.get(key)}:{value}")
    return "\n".join(lines)

# 8.元数据检索
def retrieve_hits(
    query: str,
    index: VectorStoreIndex,
    filters: dict[str, str] | None = None,
    top_k: int = 5
) -> list[NodeWithScore]:

    metadata_filters = None

    if filters:
        metadata_filters = MetadataFilters(
            filters=[
                ExactMatchFilter(
                    key=key,
                    value=value
                )
                for key, value in filters.items()
            ]
        )

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=metadata_filters
    )

    return retriever.retrieve(query)

# 9.
def retrieve(query:str,
             index:VectorStoreIndex,
             vocab:dict[str,set[str]],
             total:int,
             top_k:int=5)->list[str]:
    auto_filters=filters_from_query(query,vocab)
    lines=[
        "【场景】笨笨在找书：\n"
        f"笨笨输入：{query}\n"
        f"解析出来的过滤条件：{format_filters(auto_filters)}\n"
        f"当前数据库中有{total}本书\n"
        f"系统自动识别出了{len(auto_filters)}个过滤条件，将取{top_k}个\n"
    ]
    filterd=retrieve_hits(
        query=query,
        index=index,
        filters=auto_filters,
        top_k=top_k
    )
    lines.extend(visualize_hits(filterd))
    return lines

if __name__ == "__main__":
    # 1. 加载 JSON 数据
    records = util.load_records(
        config.CHUNK_DATA_DIR / "metadata_catalog.json"
    )
    # 2. 汇总所有 metadata 可选值
    vocab = build_vocab(records)
    # 3. 构建 TextNode
    nodes = build_metadata_nodes(records)
    # 4. 构建向量索引
    index = build_index(nodes)
    # 5. 用户查询
    query = "我想找一本作者是余华，类别为文学，在理工区，状态为在库的作品"
    # 6. 执行 metadata + 向量检索
    results = retrieve(
        query=query,
        index=index,
        vocab=vocab,
        total=len(records),
        top_k=5
    )
    # 7. 输出结果
    for r in results:
        print(r)