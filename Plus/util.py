# 工具方法文件
import json
from pathlib import Path
from Plus import config
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import Generator
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.deepseek import DeepSeek
import os
from dotenv import load_dotenv
load_dotenv()

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

# 模型调用
def create_deepseek_llm(
    temperature: float = 0.5, # 模型回答的随机性，0.0表示最确定，1.0表示最随机
    max_tokens: int = 1024, # 模型回答的最大长度，单位为token
) -> DeepSeek:
    return DeepSeek(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

llm = create_deepseek_llm()

# 单轮对话
def complete(
    prompt: str
) -> str:
    return llm.complete(prompt=prompt).text


# 单独对话(流式返回)
def stream_complete(
    prompt: str
) -> Generator[str, None, None]: # Generator[产生的数据类型, send()传入的数据类型, 最终额外return的数据类型]
    for chunk in llm.stream_complete(prompt=prompt):
        if chunk.delta:
            yield chunk.delta # yield关键字用于生成器函数中，表示生成一个值并暂停函数的执行，等待下一次迭代请求。

# 多轮对话
def chat(
    user_prompt: str,
    sys_prompt: str,
) -> str|None:
    messages = [
        ChatMessage(role = MessageRole.SYSTEM, content=sys_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    return llm.chat(messages).message.content

# 多轮对话(流式)
def stream_chat(
    user_prompt: str,
    sys_prompt: str,
) -> Generator[str, None, None]:
    messages = [
        ChatMessage(role=MessageRole.USER, content=user_prompt),
        ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
    ]
    for chunk in llm.stream_chat(messages):
        if chunk.delta:
            yield chunk.delta
