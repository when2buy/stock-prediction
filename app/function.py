from typing import Union


# 如果是经常使用的库函数，推荐加上类型注解
def add(a: Union[int, str], b: Union[int, str]) -> int:
    return int(a) + int(b)
