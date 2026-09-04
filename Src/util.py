from llama_index.core import SimpleDirectoryReader
import re
import unicodedata
from pathlib import Path
from llama_index.core import Document

import re
import unicodedata
from pathlib import Path

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


def parse_all_documents(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents


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


def clean_doc(doc):
    suffix = Path(doc.metadata.get("file_path", "")).suffix
    return Document(
        text=clean_text(doc.text, suffix),
        metadata=doc.metadata
    )


def clean_all_formats(input_dir):
    docs = parse_all_documents(input_dir) # 注意这里已经解析了Document
    cleaned_docs = []
    for i, doc in enumerate(docs):
        cleand_doc = clean_doc(doc)
        cleaned_docs.append(cleand_doc)
    return cleaned_docs
