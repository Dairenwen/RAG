from llama_index.core import SimpleDirectoryReader
import re
import unicodedata
from pathlib import Path
from llama_index.core import Document
import re
import unicodedata
from pathlib import Path
from typing import Generator
from llama_index.core.llms import ChatMessage, MessageRole
from .config import base_path
from llama_index.llms.deepseek import DeepSeek
import os
from dotenv import load_dotenv
load_dotenv()

_PPTX_TITLE_LINE = re.compile(r"^Title:\s*.+\s*$", re.MULTILINE)
_PPTX_SEPARATOR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_PPTX_SPEAKER_NOTES = re.compile(r"^\[Speaker Notes\]:\s*", re.MULTILINE)

_MD_HEADING_BOLD = re.compile(r"^#\s*\*\*(.+?)\*\*\s*$", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LEADING_HASH = re.compile(r"^#\s+", re.MULTILINE)
_INLINE_HASH_WRAP = re.compile(r"#\s*(.+?)\s*#")
_TRAILING_HASH = re.compile(r"#\s*$")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\ufeff\u200b\u200c\u200d\ufeff]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

# 解析所有文档
def parse_all_documents(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents

# 对单段文本执行通用清洗逻辑
def clean_text(text, source_suffix):
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if source_suffix.lower() in {".pptx", ".ppt", ".pptm"}:
        text = clean_ppt(text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = _MULTI_BLANK_LINES.sub("\n\n", "\n".join(lines))

    return text.strip()

# PPT专用清洗逻辑
def clean_ppt(text):
    text = _PPTX_TITLE_LINE.sub("", text)
    text = _PPTX_SEPARATOR.sub("", text)
    text = _PPTX_SPEAKER_NOTES.sub("", text)

    text = _MD_HEADING_BOLD.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)

    while True:
        cleand = _INLINE_HASH_WRAP.sub(r"\1", text)
        if cleand == text:
            break
        text = cleand

    text = _LEADING_HASH.sub("", text)
    text = _TRAILING_HASH.sub("", text)
    text = re.sub(r"\s+#\s+", "", text)

    return text.replace("#", "")

# 清洗单个文档
def clean_doc(doc):
    suffix = Path(doc.metadata.get("file_path", "")).suffix
    return Document(
        text=clean_text(doc.text, suffix),
        metadata=doc.metadata
    )

# 清洗所有文档
def clean_all_formats(input_dir):
    docs = parse_all_documents(input_dir) # 注意这里已经解析了Document
    cleaned_docs = []
    for i, doc in enumerate(docs):
        cleand_doc = clean_doc(doc)
        cleaned_docs.append(cleand_doc)
    return cleaned_docs


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
