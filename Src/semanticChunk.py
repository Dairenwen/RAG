from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List
from llama_index.core.schema import Document
import util
from llama_index.core.node_parser import SemanticSplitterNodeParser
import config


# 测试嵌入模型的方法
def embedding_similarity(
    sentence1: str,
    sentence2: str,
) -> float:
    # 1. 加载 Ollama 中的 Qwen3-Embedding
    embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:latest",
        base_url="http://localhost:11434"
    )

    # 2. 计算文本的向量
    vec1 = embed_model.get_text_embedding(sentence1)
    vec2 = embed_model.get_text_embedding(sentence2)

    # 3. 根据向量求相似度
    similarity = embed_model.similarity(vec1, vec2)

    # 4. 打印
    print(f"\n句子1: {sentence1}")
    print(f"向量1: (维度 {len(vec1)}): {vec1}")

    print(f"\n句子2: {sentence2}")
    print(f"向量2: (维度 {len(vec2)}): {vec2}")

    print(f"\n句子相似度: {similarity:.4f}")

    return similarity


# 通过嵌入模型实现语义分块
def semantic_chunk_documents(
    input_str: str,
) -> List[Document]:

    # 1. 获取清洗之后的所有文档
    documents = util.clean_all_formats(input_str)

    # 2. 加载 Ollama 中的 Qwen3-Embedding
    embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:latest",
        base_url="http://localhost:11434"
    )

    # 3. 创建语义分块器
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model
    )

    # 4. 执行语义分块
    chunks = []

    for d in documents:
        text = d.text
        if not text:
            continue
        path = d.metadata.get("file_path")

        # 对单篇文档执行语义切分
        nodes = splitter.get_nodes_from_documents(
            [
                Document(
                    text=text,
                    metadata=d.metadata
                )
            ]
        )

        # 5. 将 Node 转换成 Document
        for i, n in enumerate(nodes):
            m = dict(n.metadata)

            m.update({
                "source_file_path": path,
                "chunk_index": i,
            })

            chunks.append(
                Document(
                    text=n.text,
                    metadata=m
                )
            )

    # 6. 打印结果
    for i, chunk in enumerate(chunks):
        print(f"第{i + 1}个分块")
        print(chunk.metadata)
        print(chunk.text)
        print("-" * 50)

    return chunks

if __name__ == "__main__":
    # 示例用法
    input_dir = config.base_path / "Docs"
    semantic_chunk_documents(input_dir)