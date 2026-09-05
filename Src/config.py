from pathlib import Path
from dotenv import load_dotenv
# 获取当前文件的父目录的父目录，即项目根目录
base_path = Path(__file__).parent.parent
# 加载项目根目录下的 .env 文件
load_dotenv(base_path / ".env")