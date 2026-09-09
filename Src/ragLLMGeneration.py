import util
import time
from typing import List
from llama_index.core import Document
from config import base_path
import fixedSizedChunk
import llmChunk
import ragPromptEnhancer
import recursiveChunk
import semanticChunk
import sentenceChunk
import structureChunk

if __name__ == "__main__":
    # 1. 生成数据库
    ragPromptEnhancer.init()
    # 2. 数据清洗
    chunks: List[Document] = []
    # chunks = fixedSizedChunk.fixed_sized_chunk_documents(base_path / "Docs")
    chunks = sentenceChunk.sentence_chunk_documents(base_path / "Docs")
    # chunks = semanticChunk.semantic_chunk_documents(base_path / "Docs")
    # chunks = recursiveChunk.recursive_chunk_documents(base_path / "Docs")
    # chunks = structureChunk.structure_chunk_documents(base_path / "Docs")
    # chunks = llmChunk.llm_chunk_documents(base_path / "Docs")
    # 3. 数据入库
    ragPromptEnhancer.index_to_milvus_by_chunks(chunks)
    # 4. 提示词增强
    while True:
        question = input("请输入用户查询的问题: ").strip()
        if question.lower() in ("quit"):
            print("已经退出循环")
            break
        if not question:
            continue
        prompt = ragPromptEnhancer.get_prompt(question)
        print(prompt)
        print("======================================")
        for chunk in util.stream_complete(prompt):
            print(chunk, end="")
        print("\n======================================")