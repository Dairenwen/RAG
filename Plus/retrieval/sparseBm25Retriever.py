# 使用bm25算法来实现稀疏检索
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from Plus import util, config
import jieba

TOKEN_PATTERN = r"(?u)\b\w+\b" # 空格分开的每一段完整读取出来

def jieba_tokenize(text: str) -> str:
    return " ".join(jieba.lcut(text))

# 把json数组转换成list[TextNode]
def build_nodes(records : list[dict]) ->list[TextNode]:
    nodes : list[TextNode] = []

    for record in records:
        # 原始文本
        raw_text = f"病症：{record['disease_name']}，症状描述：{record['symptom_text']}"
        # jieba 中文分词
        text = jieba_tokenize(raw_text)
        nodes.append(
            TextNode(
                text=text,
                id = record["doc_id"],
                metadata={
                    "disease_name":record['disease_name'],
                    "raw_text": raw_text,  # 保存原文，方便后续查看
                }
            )
        )

    return nodes

# 创建稀疏检索的检索器
def create_retriever(
        nodes: list[TextNode],   # 需要建立 BM25 检索的文档节点列表
        top_k: int,              # 每次检索返回相关性最高的前 top_k 个节点
) -> BM25Retriever:
    return BM25Retriever(
        nodes=nodes,                     # 文档数据，BM25 会基于这些节点建立关键词索引
        similarity_top_k=top_k,
        skip_stemming=True,              # 跳过词干提取，适合中文文本
        token_pattern=TOKEN_PATTERN      # 指定分词/词语匹配规则
    )

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.RETRIEVAL_DATA_PATH / "retrieval.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 建立检索器
    retriever = create_retriever(nodes, 5)
    while True:
        query = input("请输入查询: ").strip()
        hits = retriever.retrieve(jieba_tokenize(query))
        for hit in hits:
            disease_name = hit.node.metadata.get("disease_name")
            score = hit.score
            print(f"【症状】：{disease_name} 【相似度】 {score:.4f}")