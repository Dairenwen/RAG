from Plus import util,config
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, FilterOperator, MetadataFilters


# 1. 把json树形结构转换成三层节点列表
def build_nodes(tree : list[dict]) -> tuple[list[TextNode], list[TextNode],list[TextNode]]:
    # 1. 第一层
    zones : list[TextNode] = []
    # 2. 第二层  专柜
    shelves: list[TextNode] = []
    # 3. 第三层   具体的商品列表
    leaves: list[TextNode] = []

    # 楼层
    for zi, zone in enumerate(tree):
        zone_id = f"z{zi}"
        zones.append(
            TextNode(
                text=f"{zone['zone']}:{zone['zone_desc']}",
                id_ = zone_id,
                metadata={
                    "zone_id":zone_id,
                    "zone":zone["zone"]
                }
            )
        )

        # 专柜层
        for si, shelf in enumerate(zone["shelves"]):
            shelf_id = f"{zone_id}_s{si}"
            shelves.append(
                TextNode(
                    text=f"{zone['zone']}/{shelf['shelf']}:{shelf['shelf_desc']}",# 它需要知道自己的父级语义环境
                    id_=shelf_id,
                    metadata={
                        "zone_id": zone_id,
                        "shelf_id": shelf_id,
                        "shelf": shelf["shelf"]
                    }
                )
            )

            # 商品层
            for bi, item in enumerate(shelf["books"]):
                leaves.append(
                    TextNode(
                        text=item["text"], # 直接将内容语义描述作为文本
                        id_=f"{shelf_id}_b{bi}",
                        metadata = {
                            "zone_id": zone_id,
                            "shelf_id": shelf_id,
                            "title": item["title"],
                            "call_no": item["call_no"]
                        }
                    )
                )
    return zones, shelves, leaves


# 2.构建三层索引
def build_index(zones:list[TextNode],shelves:list[TextNode],leaves:list[TextNode])->tuple[VectorStoreIndex,VectorStoreIndex,VectorStoreIndex]:
    embed_model=util.get_embed_model()
    zone_index=VectorStoreIndex(nodes=zones,embed_model=embed_model)
    shelf_index=VectorStoreIndex(nodes=shelves,embed_model=embed_model)
    leaf_index=VectorStoreIndex(nodes=leaves,embed_model=embed_model)
    return zone_index,shelf_index,leaf_index



# 3. 执行三层检索
def retrieve_hits(
        query:str,
        zone_index: VectorStoreIndex,
        shelf_index: VectorStoreIndex,
        leaf_index: VectorStoreIndex,
        zone_top_k: int =1,
        shelf_top_k: int =3,
        leaf_top_k: int =5,
) -> tuple[list[NodeWithScore] | NodeWithScore, list[NodeWithScore], list[NodeWithScore]]:

    # 1. 查出来的楼层
    zone_hit = zone_index.as_retriever(similarity_top_k=zone_top_k).retrieve(query)
    zone_id = [zone_hit.node.metadata["zone_id"] for zone_hit in zone_hit]

    # 2. 构造专柜查询的过滤条件
    zone_filter = MetadataFilters(
        filters=[MetadataFilter(key="zone_id", value=zone_id,operator=FilterOperator.IN)] # 当id有多个使用IN，只有一个时使用EQ
    )

    # 3. 查询专柜
    shelf_hits = shelf_index.as_retriever(similarity_top_k=shelf_top_k,filters=zone_filter).retrieve(query)

    shelf_ids = [hit.node.metadata["shelf_id"]  for hit in shelf_hits]

    # 4. 构造查询商品的过滤条件
    shelf_filter = MetadataFilters(
        filters=[MetadataFilter(key="shelf_id", value=shelf_ids,operator=FilterOperator.IN)]
    )

    # 5. 商品层元数据
    leaf_hits = leaf_index.as_retriever(similarity_top_k=leaf_top_k,filters=shelf_filter).retrieve(query)

    return zone_hit, shelf_hits, leaf_hits

# 4.对外的检索
def retrieve(query:str,
             zone_index:VectorStoreIndex,
             shelf_index:VectorStoreIndex
             ,leaf_index:VectorStoreIndex,
             zone_top_k:int=1,
             shelf_top_k:int=3,
             leaf_top_k:int=5
)->list[str]:
    zone_hit, shelf_hits, leaf_hits = retrieve_hits(
        query=query,
        zone_index=zone_index,
        shelf_index=shelf_index,
        leaf_index=leaf_index,
        zone_top_k=zone_top_k,
        shelf_top_k=shelf_top_k,
        leaf_top_k=leaf_top_k
    )

    results = [f"【场景】笨笨在逛恒隆广场，想要寻找{query}."]
    if not leaf_hits:
        results.append("未找到相关商品。")
    else:
        for i, hit in enumerate(leaf_hits):
            results.append(f"这是为您找到的第{i+1}个相关商品："
                           f"[命中商品]{hit.node.metadata.get('title', '')}，"
                           f"[命中商品号]{hit.node.metadata.get('call_no', '')}，"
                           f"[命中专柜]{hit.node.metadata.get('shelf_id', '')}，"
                           f"[命中楼层]{hit.node.metadata.get('zone_id', '')}，"
                           f"相似度:{hit.score:.5f}")
    return results

if __name__ == "__main__":
    # 1.加载数据
    records = util.load_records(config.CHUNK_DATA_DIR / "levelIndexChunk.json")
    # 2.构建节点
    zones, shelves, leaves = build_nodes(records)
    # 3.构建索引
    zone_index, shelf_index, leaf_index = build_index(zones, shelves, leaves)
    # 4.执行检索操作
    query = "雅诗兰黛持妆粉底液在哪里？"
    results = retrieve(query, zone_index, shelf_index, leaf_index, zone_top_k=1, shelf_top_k=3, leaf_top_k=5)
    for r in results:
        print(r)