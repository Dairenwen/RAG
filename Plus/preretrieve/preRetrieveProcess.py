from Plus import util,config
import jieba
import jieba.analyse as ja
import re

# 1. 加载拼写纠正的规则
def load_correction_rules() -> list[dict]:
    return util.load_records(config.PRERETRIEVE_DATA_DIR / "pinyin_corrections.json")
def load_stopwords() ->frozenset[str]:
    words = [w.strip() for w in util.load_records(config.PRERETRIEVE_DATA_DIR / "stopwords.json") if w.strip()]
    return frozenset(words) | frozenset(w.lower() for w in words if w.isascii()) # 返回一个包含原始停用词与小写英文停用词的并集

# 分词的辅助工具
def _cut(text: str) -> list[str]:
    return [t for t in jieba.cut(text) if t.strip() ]

# 判断错别字
def _match_context(text: str, pos: int, wrong: str, rule: dict) -> bool:
    prev = rule.get("prev", "")
    next_chars = rule.get("next", "")
    # 有前文要求，就检查 wrong 前面是否以 prev 结尾
    if prev:
        if not text[:pos].endswith(prev):
            return False

    # 前文检查通过后，再检查后文
    if next_chars:
        if not text[pos + len(wrong):].startswith(next_chars):
            return False

    return True

# 3. 纠正错别字
def correct_spelling(text: str, rules: list[dict]) -> str:
    out = text
    sorted_rules = sorted(rules, key=lambda  r: len(r["wrong"]), reverse=True) # 避免短规则提前把长词破坏掉

    plain_rules = [r for r in sorted_rules if "prev" not in r and "next" not in r]
    for r in plain_rules:
        out = out.replace(r["wrong"], r["right"]) # 先处理没有前后文的错别字

    ctx_rules = [r for r in sorted_rules if "prev" in r or "next" in r]
    i = 0
    while i < len(out):
        match = False
        for rule in ctx_rules:
            w, r = rule["wrong"], rule["right"]
            wl = len(w)
            if out[i:i+wl] == w and _match_context(out, i, w, rule): #如果前后文都符合并且错别字也符合替换条件
                out = f"{out[:i]}{r}{out[i+wl:]}"
                match = True
                break
        i += 0 if match else  1
    return out


# 4. 删除停用词
def remove_stopwords(text: str, stopwords:frozenset[str]) ->str:
    kept = [t for t in _cut(text.strip())  if t not in stopwords and t.lower() not in stopwords]

    if not kept:
        return ""

    if all(re.fullmatch(r"[\x00-\x7f]+", t) for t in kept): # 如果全部都是英文单词，返回空格分隔的字符串
        return " ".join(kept)
    else: # 中文词语直接拼接
        return "".join(kept)


# 5. 提取关键词
def extract_keywords(text: str, count: int =5) -> str:
    query = text.strip()
    if not query:
        return ""

    try:
        for fn in (ja.extract_tags, ja.textrank): # 先尝试使用jieba的关键词提取方法，
            # 如果没有安装jieba，则使用正则表达式提取英文单词
            tags = fn(query, topK=count)
            if tags:
                return ", ".join(tags)

    except ImportError:
        pass

    en = re.findall(r"[A-Za-z]{3,}", query)
    if en:
        return ", ".join(en[:count])

    longs = sorted({t for t in _cut(query) if len(t) >= 2}, key=len,reverse=True) # 按长度降序排列，取前count个
    return ", ".join(longs[:count])

# 6.合并
def process_steps(text: str, rules: list[dict], stopwords:frozenset[str], keyword_count:int=5) -> dict:
    corrected = correct_spelling(text, rules)
    cleaned = remove_stopwords(corrected, stopwords)
    keywords = extract_keywords(cleaned, count=keyword_count)
    return {
        "original": text,
        "corrected": corrected,
        "cleaned": cleaned,
        "keywords": keywords
    }

def format_output(result:dict):
    output = [f"【处理文本】："]

    output.extend([
        f"原始查询：{result['original']}",
        f"纠错后：{result['corrected']}",
        f"去停用词：{result['cleaned']}",
        f"关键词：{result['keywords']}"
    ])

    return "\n".join(output)


if __name__ == "__main__":
    # 1. 加载拼写纠正规则
    rules = load_correction_rules()
    # 2. 加载停用词
    stopwords = load_stopwords()
    # 3. 测试文本
    while True:
        test_text = input("请输入要处理的文本：")
        # 4. 执行处理步骤
        result = process_steps(test_text, rules, stopwords)
        # 5. 输出结果
        print(format_output(result))
