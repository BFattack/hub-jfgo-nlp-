# -*- coding: utf-8 -*-
import torch
from loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        correct, total = 0, 0
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data

            with torch.no_grad():
                # NER模型的输出形状: (batch_size, seq_len, num_labels)
                output = self.model(input_ids)[0]
                pred_results = torch.argmax(output, dim=-1)

            # 计算准确率，需要mask掉 -100 的部分
            correct += ((pred_results == labels) & (labels != -100)).sum().item()
            total += (labels != -100).sum().item()

        acc = correct / (total + 1e-9)
        self.logger.info("验证集 Token 级准确率：%f" % acc)
        self.logger.info("--------------------")
        return acc
