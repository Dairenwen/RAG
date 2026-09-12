from typing import List
from Plus import util,config
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

# 1.进行数据拼装，构造父块
def parent_card_text(child_text:str,card_fields:dict[str,str]) -> str:
    lines = [f"【三国演义 · 故事卡片】{child_text}"]
    for key, value in card_fields.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

# 2.把json数组转换为List[TextNode]
def build_build_nodes(records:list[dict])-> list[TextNode]:
    nodes = []
    for record in records:
        parent_id = record["parent_id"]
        child_text = record["child_text"]
        card_fields = record["card_fields"]
        full_text=parent_card_text(child_text,card_fields)
        nodes.append(TextNode(
            text=full_text,
            id=f"{parent_id}::c0",# 这里的id是父块id加上一个后缀，表示这是一个子块
            metadata={
                "parent_id": parent_id,        # 小块，用于 Embedding
                "child_text": child_text,
                "parent_full_text": full_text, # 大块，用于送给 LLM
            }
        ))
    return nodes

# 3.构建索引
def build_index(nodes:list[TextNode]) -> VectorStoreIndex:
    # 1.获取嵌入模型
    embed_model=util.get_embed_model()
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

# 4.执行检索操作
def retrieve(query:str,index:VectorStoreIndex,top_k:int=5)->list[str]:
    # 1.根据已有的索引直接去查询
    hits=index.as_retriever(similarity_top_k=top_k).retrieve(query)
    if not hits:
        return "没有找到相关的内容"

    returnlist=[]
    # 2.选择相似度最高的数据
    for i,hit in enumerate(hits):
        returnlist.append(f"[命中子块]{hit.metadata.get("child_text","")}，"
                          f"[命中父块]{hit.node.metadata['parent_full_text']}，"
                          f"相似度:{hit.score:.5f}")
    return returnlist

if __name__=="__main__":
    # 1.加载数据
    records=util.load_records(config.CHUNK_DATA_DIR/ "parent_child_cards.json")
    # 2.构建节点
    nodes=build_build_nodes(records)
    # 3.构建索引
    index=build_index(nodes)
    # 4.执行检索操作
    query="诸葛亮的故事"
    results=retrieve(query,index)
    for result in results:
        print(result)



