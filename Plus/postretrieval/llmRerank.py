# 检索后处理   ---->    LLM 重排
import re
from typing import ClassVar

from llama_index.core import Document, PromptTemplate, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.indices.utils import default_format_node_batch_fn
from Plus import util, config
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, QueryBundle
RERANK_PROMPT = PromptTemplate(
    """
下面是候选文档：

{context_str}

用户问题：
{query_str}

请判断每个文档与用户问题的相关程度。

要求：
1. 按相关性从高到低排列；
2. 相关性分数为 1~10；
3. 不要解释；
4. 必须严格按照下面格式输出：

Doc: 1, Relevance: 9
Doc: 2, Relevance: 7
Doc: 3, Relevance: 5
"""
)


def parse_rerank_answer(answer: str, num_choices: int) -> tuple[list[int], list[float]]:
    """解析 LLM 重排结果，兼容 Markdown、中文标点等常见输出差异。

    LlamaIndex 的默认解析器会把不完全等于 ``Doc: 1, Relevance: 9`` 的行
    直接忽略；当所有行都被忽略时，LLMRerank 就会返回空列表。
    """
    print("LLM 原始回复：")
    print(answer)

    pattern = re.compile(
        r"(?:doc(?:ument)?|文档)\s*[:：#-]?\s*\*{0,2}(\d+)\*{0,2}"
        r"[\s,，;；\-—]*"
        r"(?:relevance|相关(?:性)?|评分|分数)\s*[:：]?\s*\*{0,2}"
        r"(\d+(?:\.\d+)?)\*{0,2}",
        re.IGNORECASE,
    )

    choices: list[int] = []
    relevances: list[float] = []
    seen: set[int] = set()
    for doc_number, relevance in pattern.findall(answer):
        choice = int(doc_number)
        score = float(relevance)
        if choice in seen or not 1 <= choice <= num_choices or not 1 <= score <= 10:
            continue
        seen.add(choice)
        choices.append(choice)  # LLMRerank 使用 1 开始的文档编号
        relevances.append(score)

    if not choices:
        print("警告：未能从 LLM 回复中解析出任何文档编号和相关性分数。")
    return choices, relevances


class StringContextLLMRerank(LLMRerank):
    """让自定义字符串 Prompt 在 chat LLM 上也收到 ``context_str``。"""

    max_parse_attempts: ClassVar[int] = 3

    def _get_predict_kwargs(self, nodes_batch, query_str):
        # 原实现对 chat model 只传 context_messages；但本 Prompt 需要 context_str。
        return {
            "context_str": self._format_node_batch_fn(nodes_batch),
            "query_str": query_str,
        }

    def _postprocess_nodes(self, nodes, query_bundle=None):
        """每批必须拿到完整、可解析的评分；空回复时自动重试。"""
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        initial_results = []
        for nodes_batch in self._get_node_batches(nodes):
            for attempt in range(1, self.max_parse_attempts + 1):
                raw_response = self.llm.predict(
                    self.choice_select_prompt,
                    **self._get_predict_kwargs(nodes_batch, query_bundle.query_str),
                )
                batch_results = self._parse_raw_response(raw_response, nodes_batch)
                if len(batch_results) == len(nodes_batch):
                    initial_results.extend(batch_results)
                    break
                print(
                    f"第 {attempt} 次精排回复不完整，正在重试 "
                    f"（已解析 {len(batch_results)}/{len(nodes_batch)} 条）。"
                )
            else:
                raise RuntimeError(
                    "DeepSeek 连续 3 次未返回完整、可解析的重排结果；"
                    "请检查接口服务状态后重试。"
                )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]

# 数据集
DATA_PATH = config.POST_RETRIEVAL_DATA_PATH / "reRank.json"

# 精排后保留前 N 条
TOP_N = 5

# 该模型会先生成 reasoning_content，1024 token 常会在思考阶段耗尽，
# CompletionResponse.text 因而为空。为精排单独预留足够的推理和最终答案空间。
RERANK_LLM = util.create_deepseek_llm(temperature=0.0, max_tokens=4096)

records = util.load_records(DATA_PATH)

documents = []

# 嵌入模型（双塔粗召回用）
embed_model = util.get_embed_model()


# 把 json 数组转换成 Document 列表
for item in records:
    text = f"书名：{item['book_title']}，内容：{item['book_content']}"
    metadata = {
        "book_id": item["book_id"],
        "book_title": item["book_title"],
        "book_content": item["book_content"],
    }
    documents.append(Document(text=text, metadata=metadata))

# 构建向量索引
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 创建检索器
retriever = VectorIndexRetriever(index=index, similarity_top_k=len(records))

# 模拟同学提问
query = "高三高考电磁学大题冲刺刷题"

raw_nodes: list[NodeWithScore] = retriever.retrieve(query)

print(f"初步检索结果（双塔粗召回），共{len(raw_nodes)}条")

for n in raw_nodes:
    print(f"{n.metadata['book_id']} | 分数：{n.score:.2f} |{n.metadata['book_title']}")

print("-" * 60)

print("【LLM 重排】")

# LLMRerank：大模型结合 query 理解需求，对候选逐本打分并重排
reranker = StringContextLLMRerank(
    llm=RERANK_LLM,
    top_n=TOP_N,
    choice_batch_size=5,
    choice_select_prompt=RERANK_PROMPT,
    # 显式使用字符串格式化，供 RERANK_PROMPT 的 {context_str} 使用。
    format_node_batch_fn=default_format_node_batch_fn,
    parse_choice_select_answer_fn=parse_rerank_answer,
)
# postprocess_nodes：调用大模型精排，score 会被 LLM 相关分覆盖
reranked_nodes = reranker.postprocess_nodes(
    raw_nodes,
    QueryBundle(query_str=query)
)

print(f"重排后保留 Top{TOP_N}，共{len(reranked_nodes)}条")

for i, n in enumerate(reranked_nodes, 1):
    print(
        f"[{i}] {n.metadata['book_id']} | 分数：{n.score:.2f} | "
        f"{n.metadata['book_title']} | {n.metadata['book_content']}"
    )

print("-" * 60)
