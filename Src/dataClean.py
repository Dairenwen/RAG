#正则表达式
import re
#unicode标准库
import unicodedata
from pathlib import Path
from llama_index.core import Document
from util import parse_all_documents
from config import base_path

# ---------------------- 预编译正则:PPT解析产生的结构噪声匹配规则 ----------------
# 匹配PPT解析生成的标题行:Title: xxx
_PPTX_TITLE_LINE = re.compile(r"^Title:\s*.+\s*$", re.MULTILINE)
# 匹配PPT内容分割线:连续三个及以上短横线
_PPTX_SEPARATOR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
# 匹配PPT备注前缀:[Speaker Notes]:
_PPTX_SPEAKER_NOTES = re.compile(r"^\[Speaker Notes\]:\s*", re.MULTILINE)

# ---------------------- 预编译正则:Markdown标记清理规则 ----------------------
# 匹配加粗一级标题 # **标题**
_MD_HEADING_BOLD = re.compile(r"^#\s*\*\*(.+?)\*\*\s*$", re.MULTILINE)
# 匹配行内加粗标记 **内容**
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
# 匹配行首标题符号 #
_LEADING_HASH = re.compile(r"^#\s+", re.MULTILINE)
# 匹配首尾包裹#的文本 # 内容 #
_INLINE_HASH_WRAP = re.compile(r"#\s*(.+?)\s*#")
# 匹配行尾多余#符号
_TRAILING_HASH = re.compile(r"#\s*$")

# ---------------------- 预编译正则:通用脏字符匹配规则 ----------------------
# 匹配不可见ASCII控制字符(换行､制表符除外)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 匹配零宽空白､字节序标记等肉眼不可见隐形字符
_ZERO_WIDTH = re.compile(r"[\ufeff\u200b\u200c\u200d\ufeff]")
# 匹配连续3行及以上空行,统一压缩为两段换行
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

# 对单段文本执行通用清洗逻辑
def clean_text(text, source_suffix):

    # 删除零宽字符（肉眼看不见，但可能影响文本匹配和处理）
    text = _ZERO_WIDTH.sub("", text)

    # 删除 BOM（文件开头可能存在的隐藏字符）
    text = text.replace("\ufeff", "")

    # 统一 Unicode 字符格式
    # 例如：全角字符、特殊形式字符 → 统一成标准形式
    text = unicodedata.normalize("NFKC", text)

    # 删除控制字符，避免出现不可见的异常字符
    text = _CONTROL_CHARS.sub("", text)

    # 统一换行符
    # Windows 的 \r\n、旧 Mac 的 \r 都转换成 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 如果当前文本来自 PPT 文件，额外执行 PPT 专用清洗，去掉 PPT 特有的格式和垃圾内容
    if source_suffix.lower() in {".pptx", ".ppt", ".pptm"}:
        text = clean_ppt(text)

    # 按换行符拆成一行一行
    # line.strip()：去掉每行前后的空格
    # if line.strip()：过滤掉空行
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 重新拼接文本
    # 将多个连续空行压缩成最多一个空行
    text = _MULTI_BLANK_LINES.sub("\n\n", "\n".join(lines))

    # 去掉整个文本开头和结尾的空白
    return text.strip()


# 单独处理 PPT 文件中的特殊格式，ppt会先转换为markdown格式，需要重点清理#和**等标记
def clean_ppt(text):

    # 删除 PPT 转换过程中产生的标题行
    text = _PPTX_TITLE_LINE.sub("", text)

    # 删除 PPT 中产生的分隔线
    text = _PPTX_SEPARATOR.sub("", text)

    # 删除 PPT 的演讲者备注等无关内容
    text = _PPTX_SPEAKER_NOTES.sub("", text)

    # 去掉 Markdown 标题和加粗组合中的格式符号
    # 例如：### **Redis** → Redis（具体效果取决于正则）
    text = _MD_HEADING_BOLD.sub(r"\1", text)

    # 去掉 Markdown 加粗符号
    # 例如：**Redis** → Redis
    text = _MD_BOLD.sub(r"\1", text)

    # 反复清理被 # 包裹的内容
    # 因为一次替换可能还不能完全清理干净，所以循环执行
    while True:
        cleand = _INLINE_HASH_WRAP.sub(r"\1", text)

        # 如果这一次处理前后没有变化
        # 说明已经清理完成，退出循环
        if cleand == text:
            break

        text = cleand

    # 删除文本开头多余的 #
    # 例如：### Redis → Redis
    text = _LEADING_HASH.sub("", text)

    # 删除文本结尾多余的 #
    # 例如：Redis ### → Redis
    text = _TRAILING_HASH.sub("", text)

    # 删除被空格包围的 #
    # 例如：Redis # 数据库
    text = re.sub(r"\s+#\s+", "", text)

    # 最后把剩余的 # 全部删除
    return text.replace("#", "")



# 构建清洗后新的Document对象
def clean_doc(doc):
    suffix = Path(doc.metadata.get("file_path", "")).suffix
    return Document(text=clean_text(doc.text, suffix), metadata=doc.metadata)

# 构建清洗后的Document对象
def clean_all_formats(input_dir):
    # 原始解析出来的对象
    docs = parse_all_documents(input_dir)
    cleaned_docs = []
    for i, doc in enumerate(docs):
        cleand_doc = clean_doc(doc)
        cleaned_docs.append(cleand_doc)
        print(f"\n第{i+1} 个对象|文件路径:{cleand_doc.metadata.get("file_path","")}")
        print(cleand_doc.text)
    return cleaned_docs

if __name__ == "__main__":
    # 示例用法
    input_dir = base_path / "Docs"
    clean_all_formats(input_dir)
