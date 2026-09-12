from collections.abc import Callable
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceWindowNodeParser
from Plus import util,config

DOC_REGISTRY: dict[str, dict] = {}

# 1.为句子生成序号标签
def make_labels(count:int)->list[str]:
    return [f"句子{i+1}:" for i in range(count)]

# 2.分句函数，句子已经分好，传入自定义分句器
def make_sentence_spliter(sentence:list[str])->Callable[[str],list[str]]:
    def _split(_test:str)->list[str]:
        return list(sentence) # _split的输入与输出没有关系，直接返回已经分好的句子
    return _split

# 3.把json正文文档登记
def register_documents(records: list[dict]):
    DOC_REGISTRY.clear()
    for record in records:
        DOC_REGISTRY[record["doc_id"]]={
            "title":record["title"],
            "sentences":record["sentences"]
        }

# 4.将json数据转换成textnode列表
def build_sentence_nodes(records:list[dict],window_size:int)->list[TextNode]:
    register_documents(records)
    nodes:list[TextNode]=[]

    for record in records:
        doc_id = record["doc_id"]
        title=record["title"]
        sentences=record["sentences"]
        # 创建句子的切分工具
        parser=SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            sentence_splitter=make_sentence_spliter(sentences),
        )# window_size 决定每个句子TextNode额外保存多少条前后邻句，这些窗口内容会被存到该节点的 metadata["window"] 中
        # 将句子包装成一个Document对象，并生成TextNode列表
        doc_nodes=parser.get_nodes_from_documents([Document(text="".join(sentences))])

        for i,node in enumerate(doc_nodes):
            node.metadata["doc_id"]=doc_id
            node.metadata["doc_title"]=title
            node.metadata["sentence_index"]=i
            node.id_=f"{doc_id}::s{i}"
        nodes.extend(doc_nodes)
    return nodes

# 5.构建向量索引
def build_index(nodes:list[TextNode])->VectorStoreIndex:
    embed_model=util.get_embed_model()
    return VectorStoreIndex(nodes=nodes,embed_model=embed_model)


# 6. 执行检索：用户提问后拿到相似度前n的文本，进行一个前后窗口大小的扩展
def retrieve(
    query:str,
    index:VectorStoreIndex,
    top_k:int=5
)->list[str]:
    hits=index.as_retriever(similarity_top_k=top_k).retrieve(query)

    if not hits:
        return ["没有找到相关的内容"]

    results=[]
    for hit in hits:
        doc_id=hit.metadata["doc_id"]
        doc_title=hit.metadata["doc_title"]
        center_index=hit.metadata["sentence_index"]
        window = hit.metadata["window"] # 在进行句子切分时，已经将前后邻句的内容存储在了metadata["window"]中，直接使用即可
        results.append(f"[命中句子]{center_index}.{doc_title}\n[上下文]{window}\n相似度:{hit.score:.5f}")
    return results

if __name__=="__main__":
    # 1.加载数据
    records=util.load_records(config.CHUNK_DATA_DIR/ "sentence_window_passages.json")
    # 2.构建节点
    nodes=build_sentence_nodes(records,window_size=1)
    # 3.构建索引
    index=build_index(nodes)
    # 4.执行检索操作
    query="雨夜居家读书"
    results=retrieve(query,index,top_k=5)
    for result in results:
        print(result)