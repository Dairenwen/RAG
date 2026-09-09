from pathlib import Path
from dotenv import load_dotenv
# 获取当前文件的父目录的父目录，即项目根目录
base_path = Path(__file__).parent.parent
# 加载项目根目录下的 .env 文件
load_dotenv(base_path / ".env")
# 项目1的数据集以及测试集路径
Project1_path=Path(__file__).parent.parent / "Project1"
data_path = Project1_path / "medical_QA"
dataless_path=Project1_path/"test_QA"
