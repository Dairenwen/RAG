import shutil
import time
from typing import List
import config
import milvus
from llama_index.core.schema import Document
import sentenceChunk

# 提示词的模版
PROMPT_TEMPLATE = """【角色】你是一个专业的客服/技术顾问,只基于文档回答问题｡
【任务】根据文档回答用户的问题,禁止瞎编乱造
【上下文】
{context}
【用户的问题】
{question}
【输出要求】
1. 准确､简洁,分条回答;
2. 当用户查询无关信息的时候,直接说“文档中不存在”;
3. 不要用文档以外的知识去回答｡
"""

DB_NAME = "ragPromptEnhancer"
COLLECTION = "ragPromptEnhancer"
DIMENSION = 4096

# 1. 构建向量数据库
def init() -> None:
    # 1. 判断db是否存在
    db_path = config.base_path / f"{DB_NAME}.db"
    if db_path.exists():
        shutil.rmtree(db_path)
        time.sleep(0.5)

    # 2. 创建db
    milvus.create_db(DB_NAME)
    milvus.create_collection(COLLECTION, DIMENSION)


# 2. 把知识库分块,入库
def index_to_milvus(doc_dir: str) -> int:
    # 1. 句子分块
    chunks = sentenceChunk.sentence_chunk_documents(doc_dir, max_sentences=1)
    if not chunks:
        raise ValueError(f"句子分块结果为空,请重新检查目录:{doc_dir}")

    # 2. 构造插入向量数据库的数据集
    rows = []
    for i, chunk in enumerate(chunks):
        # 3. 计算向量
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append({"id": i+1, "vector":vec, "text": chunk.text})
    # 4. 向量入库
    milvus.insert(COLLECTION, rows)
    print(f"已经完成入库:{len(rows)}, 向量的维度{DIMENSION}")
    return len(rows)

def index_to_milvus_by_chunks(chunks: List[Document]) -> int:
    rows = []
    for i, chunk in enumerate(chunks):
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append({"id": i + 1, "vector": vec, "text": chunk.text})
    milvus.insert(COLLECTION, rows)
    print(f"已经完成入库:{len(rows)}, 向量的维度{DIMENSION}")
    return len(rows)


# 3. 实现向量检索
def retrieve(question:str, top_k: int = 5)-> List[dict]:
    milvus.get_client().load_collection(collection_name=COLLECTION) # 进行查询前，一定要记住加载collection到内存中
    results = milvus.search_by_text(
        collection_name=COLLECTION,
        text=question,
        limit=top_k,
        fields=["id", "text"]
    )
    return results[0] if results else []

# 4. 实现上下文的压缩
def compress_context(texts: List[str], max_chars: int=500) -> str:
    seen = set() # 用来记录已经出现过的文本,避免重复
    parts = []
    total = 0 # 用来记录当前已经有了多少个字符

    for text in texts:
        seen.add(text)

        sep_len = 2 if parts else 0 # 计算分隔符的长度,如果是第一个文本,就不需要分隔符,否则需要加上两个换行符
        # 需要对已有长度进行判断
        if total + sep_len + len(text) > max_chars:
            # 记录当前的字符串容量
            remain = max_chars - total - sep_len

            if remain > 20: # 如果剩余的容量大于20个字符,就截取一部分文本,并加上省略号，否则没有截取的意义
                parts.append(text[:remain] + "...")
                break
            # 剩余的文本直接舍弃,不再加入
        else:
            parts.append(text)
            total += sep_len + len(text)

    return "\n\n".join(parts)

# 5. 实现上下文的过滤
def filter_chunks(hits: List[dict], min_similarity: float = 0.5) -> List[str]:
    texts = []

    for hit in hits:
        if hit.get("distance", 0.0) < min_similarity: # COSINE 指标越大越相似
            continue
        text = hit.get("entity", {}).get("text","").strip()
        if len(text) >= 10: # 加入在数据库中查询出来的内容太短,直接舍弃
            texts.append(text)

    return texts

# 6. 构造提示词
def build_prompt(context: str,question: str,) -> str:
    if not context:
        context = "(无相关的文档片段)"
    return PROMPT_TEMPLATE.format(context=context, question=question)

# 7. 根据原始文档和用户问题,生成提示词
def get_prompt(
    question: str,
    top_k: int=5,
    min_similarity: float=0.5,
    max_chars: int =500,
) -> str:
    # 提示词增强过程为:检索->过滤->压缩->构造提示词
    hits = retrieve(question, top_k=top_k)
    texts = filter_chunks(hits, min_similarity=min_similarity)
    context = compress_context(texts, max_chars=max_chars)
    prompt = build_prompt(context, question)
    return prompt

# 8. 处理方法(循环生成增强后的提示词)
def interactive():
    while True:
        question = input("请输入用户查询的问题: ").strip()
        if question.lower() in ("quit"):
            break
        if not question:
            continue
        prompt = get_prompt(question)
        print(prompt)

if __name__ == "__main__":
    init()
    index_to_milvus(config.base_path / "Docs")
    interactive()
