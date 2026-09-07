# 向量检索
import shutil
import time
import config
import milvus
from sentenceChunk import sentence_chunk_documents

# 1. 构建向量数据库
DB_NAME = "vectorSearch"
COLLECTION = "vectorSearch"
DIMENSION = 4096

def init():
    # 1. 判断db是否存在
    db_path = config.base_path / f"{DB_NAME}.db"
    if db_path.exists():
        shutil.rmtree(db_path) # 如果存在就删除,重新创建
        time.sleep(0.5)

    # 2. 创建db
    milvus.create_db(DB_NAME)
    milvus.create_collection(COLLECTION, DIMENSION)

# 2. 文本分块,分块内容逐步向量化入库(文本分块的方案->句子分块)
def index_to_milvus() -> int:
    chunks = sentence_chunk_documents(config.base_path / "Docs", 1)
    if not chunks:
        raise ValueError("分块结果为空,请检测数据集内容")
    rows = []

    # 把分好块的内容向量化,拼装成要入库的数据
    for i, chunk in enumerate(chunks):
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append(
            {
                "id": i+1,
                "vector": vec,
                "text": chunk.text
            }
        )
    milvus.insert(COLLECTION, rows)
    return len(rows)

# 3. 文本检索
def search_similar(query:str, limit: int = 5):
    milvus.get_client().load_collection(collection_name=COLLECTION)
    # 进行 Search 或 Query 前，Collection 必须处于 Loaded 状态，加载进内存中
    results = milvus.search_by_text(
        COLLECTION,
        text=query,
        limit=limit,
        fields=["id", "text"],
    )
    hits = results[0] if results else [] # results是一个二维列表,每个元素是一个查询结果列表,这里取第一个查询结果列表
    if not hits:
        print("未检索到相关结果")
        return
    print(f"\n用户查询:{query}")
    print(f"共{len(hits)} 条相似的结果")

    # hit = {
    #     "id": 3,                  # 这条数据的 ID
    #     "distance": 0.1234,       # 与查询向量的距离/相似度指标
    #     "entity": {               # 这条数据本身的字段
    #         "id": 3,
    #         "text": "Redis 是一种内存数据库..."
    #     }
    # }
    for rank, hit in enumerate(hits, start=1): # rank从1开始
        entity = hit.get("entity") # 拿出实际数据字段
        print(f"\n[{rank}] 距离 distance={hit.get('distance'):.4f}")
        print(f"id={entity.get('id', hit.get('id'))}")
        print(entity.get("text"))

# 4. 循环输入用户查询,来获取知识库内容
def search():
    print("\n向量检索就绪,请输入用户查询问题")
    while True:
        query = input("请输入查询问题:").strip()
        if query.lower() == ("quit"):
            break
        if not query:
            continue
        search_similar(query)

if __name__ == "__main__":
    init()
    index_to_milvus()
    search()