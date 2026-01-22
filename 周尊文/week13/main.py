# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import logging
import numpy as np
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)
    model = TorchModel  # 来自model.py的TokenClassification模型

    # LoRA 配置
    if config["tuning_tactics"] == "lora_tuning":
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,  # 关键修改：任务类型变为TOKEN_CLS
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],  # BERT常用target
            # modules_to_save=["classifier"] # 官方推荐方法：指定分类头不冻结
        )
        model = get_peft_model(model, peft_config)

        # 如果不使用 modules_to_save，可以用您原有的逻辑手动解冻分类头
        # 注意：AutoModelForTokenClassification 的分类层通常叫 'classifier'
        # 下面这段是为了兼容您之前的写法风格：
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    if torch.cuda.is_available():
        logger.info("gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data

            # HuggingFace的模型内部会自动计算Loss (如果传入了labels)
            # labels中的-100会自动被CrossEntropyLoss忽略
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % 10 == 0:
                logger.info("epoch %d batch %d loss %f" % (epoch, index, loss.item()))

        logger.info("epoch %d average loss: %f" % (epoch, np.mean(train_loss)))
        evaluator.eval(epoch)

    # 保存LoRA权重
    model_path = os.path.join(config["model_path"], "lora_ner.pth")
    # 仅保存可训练参数
    torch.save(
        {k: v for k, v in model.state_dict().items() if v.requires_grad},  # 这种写法更稳健
        model_path
    )


if __name__ == "__main__":
    main(Config)
