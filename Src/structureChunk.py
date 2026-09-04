import re
import csv
from typing import List
from llama_index.core.schema import Document

import util
import config
from Src.config import base_path

max_chunk_chars = 1500

# PDF 页内次级结构切分点:编号问题(1.)或项目符号(• ◦ -)前
_PDF_SUBSECTION_RE = re.compile(r"(?=\n(?:\d+\.|•|◦|\-)\s*)")

# 从ppt的解析结果提取文本
def _extract_section_text(section: dict) -> str:
    return section.get("content")

# 处理PPT类型文档分块
def _chunk_ppt_document(
    d: Document
) -> List[Document]:
    sections = d.metadata.get("text_sections")
    slide_title = d.metadata.get("title")

    texts = [_extract_section_text(s) for s in sections]
    texts = [t for t in texts if t]

    full_text = "\n".join(texts)
    if len(full_text) <= max_chunk_chars:
        meta = {
            "chunk_index": 0
        }
        return [Document(text=full_text, metadata=meta)]












# 基于文档结构的分块
def structure_chunk_documents(
    input_str: str,
) -> List[Document]:
    # 1. 获取清洗之后的数据
    documents = util.clean_all_formats(input_str)

    # 2. 逐一分块
    chunks = []
    for d in documents:
        doc_chunks = _chunk_single_document(d)
        chunks.extend(doc_chunks)

    # 3. 测试输出
    for i, chunk in enumerate(chunks):
        print(f"第{i+1}个分块")
        print(chunk.metadata)
        print(chunk.text)

    return chunks

if __name__ == "__main__":
    structure_chunk_documents(
        base_path/"Docs"
    )
