import re

_PUNCT = re.compile(r"[\s，。！？；：、,.!?;:\"'“”‘’()（）\[\]【】]")


def normalize(text: str) -> str:
    """去空白与标点，转小写，用于精确去重。

    只做确定性规范化；语义近似去重不在本章范围。
    """
    return _PUNCT.sub("", text).strip().lower()
