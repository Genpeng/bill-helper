# _*_ coding: utf-8 _*_

"""
Bill Classifier.

Author: Genpeng Xu
"""

import joblib
from typing import Union, List

# Own customized variables & modules
from bill_helper.tokenizer import MyTokenizer
from bill_helper.global_variables import (T1_VECTORIZER_FILEPATH,
                                          T1_MODEL_FILEPATH,
                                          LABEL_2_TYPE_DICT_FILEPATH)


class BillClassifier(object):
    def __init__(self):
        self._tokenizer = MyTokenizer()
        self._vectorizer = joblib.load(T1_VECTORIZER_FILEPATH)
        self._model = joblib.load(T1_MODEL_FILEPATH)
        self._label_2_type = joblib.load(LABEL_2_TYPE_DICT_FILEPATH)

    def _classify(self, texts: List[str]) -> List[int]:
        texts_segmented = [self._tokenizer.segment(text) for text in texts]
        return list(self._model.predict(self._vectorizer.transform(texts_segmented)))

    def classify_bill(self, texts: List[str]) -> List[str]:
        labels = self._classify(texts)
        return [self._label_2_type[label] for label in labels]


if __name__ == "__main__":
    texts = [
        "零星砌砖 1.LC15陶粒混凝土填充层3.钢筋混凝土楼板扫水泥浆一道 4.部位:沉箱",
        "砌块墙 1.砌块品种、规格、强度等级:蒸压加气混凝土砌体 3.砂浆强度等级:预拌水泥砂浆M5.0 4.部位:变形缝"
    ]  # types = ['围墙', '砌体及二次结构']
    clf = BillClassifier()
    types = clf.classify_bill(texts)
    print(types)