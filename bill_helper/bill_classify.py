# _*_ coding: utf-8 _*_

"""
Classify bill by using machine learning algorithms.

Author: Genpeng Xu
"""

from typing import List, Union

# Own customized modules
from bill_helper.classifier import BillClassifier

# global variables needed
clf = BillClassifier()


def classify_bill(texts: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(texts, str):
        texts = [texts]
    types = clf.classify_bill(texts)
    return types if len(types) > 1 else types[0]


if __name__ == '__main__':
    texts = [
        "零星砌砖 1.LC15陶粒混凝土填充层3.钢筋混凝土楼板扫水泥浆一道 4.部位:沉箱",
        # "砌块墙 1.砌块品种、规格、强度等级:蒸压加气混凝土砌体 3.砂浆强度等级:预拌水泥砂浆M5.0 4.部位:变形缝",
    ]  # types = ['围墙', '砌体及二次结构']
    types = classify_bill(texts)
    print(types)
