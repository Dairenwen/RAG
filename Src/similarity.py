from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.embeddings.ollama import OllamaEmbedding

import config

def embedding_similarity(
    sentence1: str,
    sentence2: str,
    mode: SimilarityMode = SimilarityMode.DEFAULT # 相似度计算模式，默认使用余弦相似度
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
    similarity = embed_model.similarity(vec1, vec2, mode=mode)

    # 4. 打印
    print(f"\n句子1: {sentence1}")
    print(f"向量1: (维度 {len(vec1)}): {vec1}")

    print(f"\n句子2: {sentence2}")
    print(f"向量2: (维度 {len(vec2)}): {vec2}")

    print(f"\n句子相似度: {similarity:.4f}")

    return similarity

if __name__ == "__main__":
    sentence1="爸爸的妈妈叫奶奶"
    sentence2="妈妈的爸爸叫外公"
    print(embedding_similarity(sentence1, sentence2, mode=SimilarityMode.DEFAULT)) # 余弦相似度
    print(embedding_similarity(sentence1, sentence2, mode=SimilarityMode.DOT_PRODUCT)) # 点积相似度
    # LlamaIndex中为了把数值排序规则统一起来,改成了结果取负数,这样越相似得分越大,越不相似得分越小
    print(embedding_similarity(sentence1, sentence2, mode=SimilarityMode.EUCLIDEAN)) # 欧几里得相似度