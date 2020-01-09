# _*_ coding: utf-8 _*_

"""
Find top k most similar texts (bills).

Author: Genpeng Xu
"""

import pandas as pd
from typing import Union, List

# Own customized modules
from core.searcher import BillSearcher

# global variables needed
searcher = BillSearcher()


def find_k_nearest_bills(query_texts: Union[str, List[str]], k: int = 5) -> List[pd.DataFrame]:
    return searcher.find_k_nearest_bills(query_texts, k)


def test():
    query_texts = [
        "空心砖墙 1、砖品种、规格、强度等级：蒸压加气混凝土砌块 2、砂浆强度等级：M5水泥石灰砂浆 3、墙体厚度：200厚砖内墙 m3",  # id: 0
        "直形墙 1、C40普通商品混凝土20石 m3",  # id: 5
        "木质防火门 1、木质乙级防火门 2、门窗五金及油漆 3、详见门窗大样 m2",  # id: 57
        "配电箱 1、 配电箱安装 AW1~7 台",  # id: 68
        "现浇构件钢筋,(1)钢筋种类、规格:箍筋圆钢制安Φ10以内,t",  # id: 1201
    ]
    k = 5
    ans = find_k_nearest_bills(query_texts, k)


if __name__ == "__main__":
    test()