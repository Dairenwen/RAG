import os

from pymilvus import MilvusClient
from .util import base_path
from typing import List,Dict,Any,Optional
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:latest",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

# 1. 连接milvus
def connect(dbpath : str) -> MilvusClient:
    global client
    client = MilvusClient(str(dbpath))
    return client


# 2. 获取client
def get_client() -> MilvusClient:
    return client

# 3. 创建db,类比mysql建库,其实就是创建一个文件夹,里面存放collection的文件夹
def create_db(dbname : str) -> MilvusClient:
    return connect(base_path / f"{dbname}.db")

# 4. 切换db use_db
def use_db(dbname : str) -> MilvusClient:
    return connect(base_path / f"{dbname}.db")

# 5. 列出所有的db
def list_dbs() -> List[str]:
    return [p.stem for p in base_path.glob("*.db")]

# 6. 创建collection，类比mysql建表，collection是milvus中存储向量的最小单位，需要指定向量的维度dimension
def create_collection(name: str, dimension: int) -> None:
    client = get_client()
    if client.has_collection(name):
        client.drop_collection(collection_name=name)
    client.create_collection(collection_name=name, dimension=dimension)

# 7. 删除collection
def drop_collection(name:str) -> None:
    get_client().drop_collection(collection_name=name)


# 8. 插入数据，类比mysql的insert into table values(...)，rows是一个字典列表，每个字典表示一行数据，键是字段名，值是字段值
# 具体的字段名和字段值需要根据实际情况来定，返回值是一个字典，包含插入的结果信息
def insert(collection_name: str, rows: List[Dict[str, Any]]) -> Dict:
    return get_client().insert(collection_name=collection_name, data=rows)

# 9. 查询数据，ids为需要查询的向量id列表，fields为需要返回的字段列表，如果为空则只返回id和vector字段
def get_by_ids(collection_name: str, ids: List[int], fields: Optional[List[str]] = None) -> List[Dict]:
    return get_client().get(collection_name=collection_name, ids=ids, output_fields=fields)


# 工具：获得一段文本的向量
def get_text_vector(text: str) -> List[float]:
    vector = embed_model.get_text_embedding(text)
    return vector

# 10. 向量查询
#  Collection、查询文本、返回数量和返回字段，
#  先将文本转为 Embedding 向量，再在 Milvus 中进行相似度检索并返回 Top-K 结果。
def search_by_text(
    collection_name:str,
    limit: int=5, # 默认返回五条最相似的结果
    text: str="",
    fields: Optional[List[str]] = None, 
) -> List[List[dict]]:
    vectors = [embed_model.get_text_embedding(text)]
    return get_client().search(
        collection_name=collection_name,
        data=vectors,
        limit=limit,
        output_fields=fields,
    )


if __name__ == "__main__":
    text = "这是一个测试文本。"
    vector = get_text_vector(text)
    print(f"文本: {text}")
    print(f"向量维度: {len(vector)}")
    print(f"向量: {vector}")
    # 测试连接
    create_db("test_milvus")
    create_collection("test_collection", 4096)
    print(insert("test_collection", [{"id": 1, "vector": vector}]))
    print(get_by_ids("test_collection", [1]))
    print(search_by_text("test_collection", limit=5, text="这是一个测试文本。"))
