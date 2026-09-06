import util
from llama_index.core import Document
from typing import List
import re
from Src.config import base_path

# llm返回的是字符串，需要特殊标识
CHUNK_DELIMITER = "===CHUNK==="

# 系统提示词
SYSTEM_PROMPT = f"""你是RAG知识库文档分块助手｡ 请把用户提供的文档切分为多个语义完整､适合向量检索的分块｡
要求:
1. 每个分块围绕一个独立的主题
2. 尽量不要拆分段落,不要拆分句子
3. 保持原始文本的语义不变
4. 各个分块之间用{CHUNK_DELIMITER}分隔开
5. 只输出分块的正文,不要额外的信息
"""

# 解析LLM返回的文本
def _parse_llm_chunks(response :str) -> List[str]:
    parts = re.split(fr"\n?{CHUNK_DELIMITER}\n?", response.strip()) # 使用正则表达式分割文本
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks

# 调用LLM
def _llm_split_text(text: str, max_chunk_chars:int) -> List[str]:
    USER_PROMPT = (
        f"请将以下文档分块,每个块不超过{max_chunk_chars}字符:\n\n"
        f"{text}"
    )
    all_chunks: List[str] =[]

    result = util.chat(USER_PROMPT, SYSTEM_PROMPT)
    all_chunks.extend(_parse_llm_chunks(result))

    return all_chunks

# LLM分块
def llm_chunk_documents(
    input_dir:str,
    max_chunk_chars: int=500
) -> List[Document]:
    # 1. 获取清洗后的数据
    documents = util.clean_all_formats(input_dir)

    # 2. 逐一执行LLM分块操作
    chunks: List[Document] = []
    for d in documents:
        text = d.text
        # 3. 调用LLM来处理分块
        text_chunks = _llm_split_text(text, max_chunk_chars)
        # 4. 修改元数据
        for i, chunk_text in enumerate(text_chunks):
            m=dict(d.metadata)
            m.update({
                "chunk_index":i
            })
            chunks.append(Document(text=chunk_text, metadata=m))

        # 测试打印输出
        for i, chunk in enumerate(chunks):
            print(f"第{i+1}个分块")
            print(chunk.metadata)
            print(chunk.text)

    return chunks

if __name__ == "__main__":
    llm_chunk_documents(base_path/"Docs")