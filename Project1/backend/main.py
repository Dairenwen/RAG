from pathlib import Path
from typing import Any, Dict, Generator, List
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Src import milvus
from Src.config import Project1_path
from Src.util import stream_complete
import time


EMBED_BATCH_SIZE = 100 # 每次批处理的文本数量
INSERT_BATCH_SIZE = 100 # 每次批处理的插入数量
COLLECTION="medical_qa"
DIMENSION=4096
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Medical RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

# 1. 读取csv并解析
def load_medical_qa(data_dir: str) -> List[Dict[str, str]]:
    # 1. 读取目录下面所有的csv文件,每行是一条回答
    csv_files = list(Path(data_dir).glob("*.csv")) # 收集csv文件
    if not csv_files:
        raise FileNotFoundError(f"当前目录{data_dir} 里面没有csv文件")

    # 2. 存放解析后的问答列表
    qa_list: List[Dict[str, str]] = []
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]

    # 3. 循环处理csv文件
    for csv_file in csv_files:
        df = None
        for enc in encodings: # 尝试不同的编码格式去打开csv文件
            try:
                df = pd.read_csv(
                    csv_file,
                    header=None,
                    names=["question", "answer"],
                    encoding=enc,
                )
                break
            except UnicodeError:
                continue
        if df is None: # 所有的编码格式去打开csv文件都是失败的
            raise ValueError(f"无法读取文件 {csv_file}，请检查文件编码格式。")

        df = df.fillna("")
        for i, row in df.iterrows():
            question=str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            if question:
                qa_list.append(
                    {
                        "question":question,
                        "answer":answer,
                    }
                )

    if not qa_list:
        raise ValueError(f"目录 {data_dir} 中间没有有效的医疗问答数据")
    return qa_list

# 2.创建向量数据库
def init_medical_db() -> None:
    client = milvus.connect(Project1_path / "medical.db")
    if not client.has_collection(COLLECTION):
        milvus.create_collection(COLLECTION, DIMENSION)

# 3.数据向量化
def build_milvus_rows(
    qa_list: List[Dict[str, str]],
    embed_bath_size: int=EMBED_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    # 1. 只转换question
    questions = [qa["question"] for qa in qa_list]
    # 2. 记录所有的向量数组
    vectors: List[List[float]] = []

    # 3. 按照批处理来向量化
    for start in range(0, len(questions), embed_bath_size):
        batch_texts = questions[start: start+embed_bath_size]
        batch_vectors = milvus.embed_model.get_text_embedding_batch(texts=batch_texts)
        vectors.extend(batch_vectors) # 注意是extend而不是append
        # 看向量化的进度
        print(f"已经执行完向量化 {min(start + embed_bath_size, len(questions))} / {len(questions)}条")

    # 4. 处理向量化结果
    if len(vectors) != len(qa_list):
        raise ValueError(
            f"向量数 {len(vectors)} 与问答数 {len(qa_list)} 数目不一致"
        )

    # 5. 构建待插入的数据
    rows: List[Dict[str, Any]] = []
    for i, (qa, vector) in enumerate(zip(qa_list, vectors), start=1):
        text = f"问:{qa['question']}\n答:{qa['answer']}"
        rows.append(
            {
                "id":i,
                "vector":vector,
                "text": text,
                "question":qa["question"],
                "answer":qa["answer"],
            }
        )
    print(f"构造待插入数据 {len(rows)}条")
    return rows


# 4. 向量化的数据插入向量数据库
def insert_data(
    rows: List[Dict[str, Any]],
    batch_size: int = INSERT_BATCH_SIZE,
) -> int:
    total = len(rows)
    inserted = 0

    for batch_idx, start in enumerate(range(0, total, batch_size)) : # 采用批处理
        batch = rows[start: start + batch_size]
        milvus.insert(COLLECTION, batch)
        inserted += len(batch)
        print(f"已经入库 {inserted}/{total} 条，进度为 {inserted/total:.2%}")
        if start + batch_size < total: # 减轻milvus的压力，避免一次性插入过多数据
            time.sleep(0.5)

    print(f"已经完成入库:{inserted} 条, 总共 {total} 条")
    return inserted


# 5. 数据检索
def search_medical(question: str, top_k: int = 5) -> List[dict]:
    milvus.get_client().load_collection(collection_name=COLLECTION)  # 进行查询前，一定要记住加载collection到内存中
    results = milvus.search_by_text(
        collection_name=COLLECTION,
        text=question,
        limit=top_k,
        fields=["id", "text", "question", "answer"],
    )
    return results[0] if results else []

# 6. 提示词增强
def build_medical_prompt(question: str, top_k: int = 5, max_chars: int = 4000) -> str:
    hits = search_medical(question, top_k=top_k)
    context_parts = []
    for hit in hits:
        entity = hit.get("entity", {})
        question_text = str(entity.get("question", "")).strip()
        answer_text = str(entity.get("answer", "")).strip()
        text = str(entity.get("text", "")).strip()
        context_parts.append(text or f"问:{question_text}\n答:{answer_text}")
    context = "\n\n".join(context_parts)[:max_chars]
    if not context:
        context = "(无相关的文档片段)"
    return (
        "【角色】你是一个专业的医疗客服，只基于文档回答问题。\n"
        "【任务】根据文档回答用户的问题，禁止瞎编乱造。\n"
        f"【上下文】\n{context}\n"
        f"【用户的问题】\n{question}\n"
        "【输出要求】\n"
        "1. 准确、简洁，分条回答；\n"
        "2. 当用户查询无关信息时，直接说“文档中不存在”；\n"
        "3. 不要使用文档以外的知识回答。"
    )

# 7. LLM生成
def main(message: str) -> Generator[str, None, None]:
    message = message.strip()
    if not message:
        raise ValueError("message cannot be empty")
    init_medical_db()
    prompt = build_medical_prompt(message)
    yield from stream_complete(prompt)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    def generate() -> Generator[str, None, None]:
        try:
            for chunk in main(request.message):
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'message': f'服务暂时不可用：{exc}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_PATH / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_PATH), name="frontend")
