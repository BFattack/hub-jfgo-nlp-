# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
from config import Config


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.label_map = config["label_map"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                text_list = line["text"]
                label_list = line["labels"]

                self.process_sentence(text_list, label_list)
        return

    def process_sentence(self, text_list, label_list):
        # 使用tokenizer处理，is_split_into_words表示输入已经是切分好的列表
        inputs = self.tokenizer(
            text_list,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()

        word_ids = inputs.word_ids()
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # 特殊符号 [CLS], [SEP] 或 Padding
                label_ids.append(-100)
            else:
                # 获取对应单词的标签
                label_str = label_list[word_idx]
                label_ids.append(self.label_map.get(label_str, 0))

        # 截断或补齐label_ids以匹配max_length
        if len(label_ids) > self.config["max_length"]:
            label_ids = label_ids[:self.config["max_length"]]
        else:
            # 这里的 -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index
            label_ids += [-100] * (self.config["max_length"] - len(label_ids))

        self.data.append([input_ids, torch.LongTensor(label_ids)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
