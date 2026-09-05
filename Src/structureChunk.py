import re
import csv
from typing import List
from llama_index.core.schema import Document
from pathlib import Path
import util
import config
from Src.config import base_path

max_chunk_chars = 1500 # 设置最大的分块长度为1500个字符

# 从ppt的解析结果提取文本
def _extract_section_text(section: dict) -> str|None:
    return section.get("content")

# 处理PPT类型文档分块
def _chunk_ppt_document(
    d: Document
) -> List[Document] | None:
    sections = d.metadata.get("text_sections") # 获取PPT的文本结构
    slide_title = d.metadata.get("title") # 获取PPT的标题
    # ppt清洗之后的结构为：
    # [
    #     {
    #         "title": "Slide 1 Title",
    #         "content": "Slide 1 Content"
    #     },
    #     {
    #         "title": "Slide 2 Title",
    #         "content": "Slide 2 Content"
    #     },
    #     ...
    # ]
    texts = [_extract_section_text(s) for s in sections]
    texts = [t for t in texts if t] # 过滤掉空文本

    full_text = "\n".join(texts)
    if len(full_text) <= max_chunk_chars:
        meta = {
            "chunk_index": 0
        }
        return [Document(text=full_text, metadata=meta)]

    else:
        # 当前页超过了最大长度,需要进行分块
        chunks = []
        for i, group in enumerate(full_text):
            chunk_text = group
            if slide_title and slide_title not in chunk_text: # 确保每一个 PPT 分块里都带有当前页面的标题，避免分块以后丢失上下文
                chunk_text = f"{slide_title}\n{chunk_text}"

            meta = {
                "section_index": i
            }
            chunks.append(Document(text=chunk_text, metadata=meta))

        return chunks

# 按照文档结构对pdf类型文档进行分块
def _chunk_pdf_document(
    d: Document
) -> List[Document]:
    text = d.text
    if not text:
        return []

    page_label = d.metadata.get("page_label") # 提取当前页的页码信息
    base_meta = {
        "page_label": page_label
    }

    if len(text) <= max_chunk_chars:
        meta = dict(base_meta)
        meta["chunk_index"] = 0
        return [Document(text=text, metadata=meta)]

    else:
        # 当前页超过了最大长度,需要进行分块，先按照换行后的编号 `1.`、`2.`、`3.`，
        # 以及 `●`、`。`、`-` 这些结构标记进行切分。
        parts = [p.strip() for p in r"(?=\n(?:\d+\. |●|。|\-)\s*)".split(text) if p.strip()]
        if len(parts) <= 1: # 如果没有找到结构标记，则按照空行进行切分
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # 转换成分块
        chunks = []
        for i, part in enumerate(parts):
            meta = dict(base_meta)
            meta.update(
                {
                    "chunk_index": i
                }
            )
            chunks.append(Document(text=part, metadata=meta))

        return chunks

# 获取csv文件的头
def _read_csv_header(path : str) -> str:
    with open(path, newline="", encoding="utf‑8") as f:
        row = next(csv.reader(f), None) # 读取第一行获取表头
        return ",".join(row)

# 处理csv类型文件
def _chunk_csv_document(
    d: Document
)->List[Document]:
    text = d.text
    if not text:
        return []

    lines = [ line.strip() for line in text.splitlines() if line.strip()] # 按照行分割并去掉空行
    path = d.metadata.get("file_path")
    header = _read_csv_header(path)
    data_lines = lines[1:] # 去掉表头行

    chunks = []
    for i, line in enumerate(data_lines):
        chunk_text = f"{header}\n{line}" # 每个分块都包含表头和当前行数据，避免丢失上下文
        meta = {
            "chunk_index": i
        }
        chunks.append(Document(text=chunk_text, metadata=meta))

    return chunks


# 单个文档分块
def _chunk_single_document(
    d: Document
)->List[Document]:
    suffix = Path(d.metadata.get("file_path", "")).suffix.lower()


    # 1. 处理ppt类型的文档
    if suffix == ".ppt" or suffix == ".pptx" or suffix == ".pptm":
        return _chunk_ppt_document(d)
    # 2. 处理pdf类型的文档
    if suffix == ".pdf":
        return _chunk_pdf_document(d)
    # 3. 处理csv类型的文档
    if suffix == ".csv":
        return _chunk_csv_document(d)
    # 4. txt类型文档，由于txt文档没有结构信息，不适合进行结构分块，因此直接返回原文档，建议使用固定大小分块或句子分块方法
    return [d]



# 基于文档结构的分块
def structure_chunk_documents(
    input_str: str,
) -> List[Document]:
    # 1. 获取清洗之后的数据
    documents = util.clean_all_formats(input_str)

    # 2. 逐一分块
    chunks : List[Document] = []
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
