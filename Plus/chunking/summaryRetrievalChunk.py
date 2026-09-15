from llama_index.core.schema import TextNode
from Plus import util,config
from llama_index.core import Document, VectorStoreIndex


# 1.json数组转换为list[TextNode]
def build_summary_nodes(records:list[dict])->list[TextNode]:
    nodes:list[TextNode]=[]
    for record in records:
        topic_id = record["topic_id"]
        title=record["title"]
        original_text=record["original_text"]
        summary=record["summary"]
        nodes.append(TextNode(
            text=summary,
            id=f"{topic_id}::summary",
            metadata={
                "topic_id": topic_id,
                "topic_title": title,
                "original_text": original_text,
                "summary_text": summary,
            }
        ))
    return nodes

# 2.构建索引
def build_index(nodes:list[TextNode],embed_model)->VectorStoreIndex:
    embed_model=util.get_embed_model()
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

# 3.执行检索操作
def retrieve(query:str,index:VectorStoreIndex,top_k:int=5)->list[str]:
    hits=index.as_retriever(similarity_top_k=top_k).retrieve(query)
    if not hits:
        return ["没有找到相关的内容"]

    returnlist=[]
    for i,hit in enumerate(hits):
        returnlist.append(f"[命中主题]{hit.metadata.get('topic_title','')}，"
                          f"[命中摘要]{hit.node.metadata['summary_text']}，"
                          f"[原始文档]{hit.node.metadata['original_text']}，"
                          f"相似度:{hit.score:.5f}")
    return returnlist

if __name__=="__main__":
    # 1.加载数据
    records=util.load_records(config.CHUNK_DATA_DIR/ "summary_retrieval_topics.json")
    # 2.构建节点
    nodes=build_summary_nodes(records)
    # 3.构建索引
    index=build_index(nodes,embed_model=util.get_embed_model())
    # 4.执行检索操作
    query="三国演义的故事"
    results=retrieve(query,index,top_k=5)
    for r in results:
        print(r)