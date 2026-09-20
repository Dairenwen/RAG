# 配置文件
from pathlib import Path
# 数据路径
CHUNK_DATA_DIR=(Path(__file__).parent
    / "chunking"
    / "data"
)
PRERETRIEVE_DATA_DIR=(Path(__file__).parent
    / "preretrieve"
    / "data"
)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen3-embedding:latest"