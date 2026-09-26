from llama_index.core.schema import NodeWithScore  # 实现混合融合
from Plus import util,config
# 引入稀疏检索相关的内容
from sparseBm25Retriever import  build_nodes, create_retriever as create_sparse_retriever

from vectorRetriever import create_retriever as create_vector_retriever


# 1. 实现分数融合
def rrf_merge(hit_list : list[list[NodeWithScore]], top_k:int, k: int=60) ->list[NodeWithScore]:
    # 1. 记录每个数据的得分
    scores: dict[str, float] = {}
    # 2. 记录每个数据对应的原始节点，方便后续进行结果封装
    node_map :dict[str, NodeWithScore] = {}
    # 3. 执行遍历操作
    for hits in hit_list:
        for rank , hit in enumerate(hits):
           node_id = hit.node.node_id
           scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k+rank+1) # 出现在两个检索器中，得分会累加
           node_map.setdefault(node_id, hit)
    # 4. 存放合并结果
    merged: list[NodeWithScore] = []

    # 5. 得分的排序
    for node_id, score in sorted(scores.items(), key=lambda x: x[1],reverse=True)[:top_k]:
        merged.append(NodeWithScore(node=node_map[node_id].node, score=score))
    return merged


# 这里实现真正的并行检索
from concurrent.futures import ThreadPoolExecutor
# 对外提供的混合检索方法
def hybrid_search(query: str, sparse_retriever, vector_retriever, top_k: int):
    # 两路检索并行执行
    with ThreadPoolExecutor(max_workers=2) as executor:
        sparse_future = executor.submit(sparse_retriever.retrieve, query)
        vector_future = executor.submit(vector_retriever.retrieve, query)
        sparse_hits = sparse_future.result()
        vector_hits = vector_future.result()

    # RRF 融合
    hybrid_hits = rrf_merge(
        [sparse_hits, vector_hits],
        top_k
    )

    return sparse_hits, vector_hits, hybrid_hits

# 格式化查询结果列表
def format_hits(hits : list[NodeWithScore]) ->list[str]:
    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        disease_name = hit.node.metadata.get("disease_name")
        score = hit.score
        lines.append(f" [{i}] {disease_name} | 得分 {score:.4f}")
    return lines

# 对外提供的打印方法
def print_result(query: str, sparse_hits, vector_hits, hybrid_hits):
    print(f"查询：{query}")
    print(f"【第一路.稀疏检索】")
    print("\n".join(format_hits(sparse_hits)))
    print(f"【第二路.稠密检索】")
    print("\n".join(format_hits(vector_hits)))
    print(f"【混合检索】")
    print("\n".join(format_hits(hybrid_hits)))

if __name__ == "__main__":
    # 1. 获取数据集
    records = util.load_records(config.RETRIEVAL_DATA_PATH / "retrieval.json")
    # 2. json数组转换成node列表
    nodes = build_nodes(records)
    # 3. 稀疏检索器
    sparse_retriever = create_sparse_retriever(nodes, 5)
    # 4. 稠密检索器
    vector_retriever =create_vector_retriever(nodes).as_retriever(similarity_top_k = 5)
    while True:
        query = input("请输入查询: ").strip()
        sparse_hits, vector_hits, hybrid_hits =hybrid_search(
            query, sparse_retriever, vector_retriever, 5
        )
        print_result(query, sparse_hits, vector_hits, hybrid_hits)