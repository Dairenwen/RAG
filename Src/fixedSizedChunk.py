from util import parse_all_documents, clean_text,clean_all_formats
from llama_index.core import Document
from typing import List
from config import base_path


# 按照固定大小分块
def fixed_size_chunk_documents(
    input_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    chunks = []
    step = chunk_size - chunk_overlap
    # 获取清洗后的文档
    documents = clean_all_formats(input_dir)

    for d in documents:
        text = d.text
        path = d.metadata.get("file_path")
        start = 0 #分块在原文中的起始位置
        idx = 0 # 当前文档的分块序号

        while start < len(text):
            end = min(start + chunk_size, len(text)) #当前分块结束位置
            # 复制原始文档
            m = dict(d.metadata)
            m.update({
                "source_file_path":path,
                "chunk_index":idx,
                "chunk_start": start,
                "chunk_end": end,
            })
            chunks.append(Document(text=text[start:end], metadata=m))

            if end >= len(text):
                break

            start = start + step
            idx = idx + 1

    for i, d in enumerate(chunks):
        print(f"第{i+1}个分块")
        print(d.metadata)
        print(d.text)

    return chunks


if __name__ == "__main__":
    # 示例用法
    input_dir = base_path/"Docs"
    fixed_size_chunk_documents(input_dir, chunk_size=500, chunk_overlap=50)
