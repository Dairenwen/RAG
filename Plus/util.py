# 工具方法文件
import json
from pathlib import Path
from Plus import config
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


# 1. 创建并返回本机 Ollama 向量模型实例
def get_embed_model() -> BaseEmbedding:
    return OllamaEmbedding(
        model_name=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

# 2. 从 JSON 文件中读取数据
def load_records(path: Path) -> list[dict]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )