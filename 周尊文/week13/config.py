# -*- coding: utf-8 -*-

Config = {
    "model_path": "output_ner",
    "train_data_path": "ner_train.json",  
    "valid_data_path": "ner_valid.json",
    "vocab_path": "chars.txt", 
    "model_type": "bert",
    "max_length": 100,
    "batch_size": 16,
    "epoch": 10,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "pretrain_model_path": "bert-base-chinese", 
    "seed": 987,
    "tuning_tactics": "lora_tuning",
    "class_num": 9, 
    "label_map": {
        "O": 0,
        "B-PER": 1, "I-PER": 2,
        "B-LOC": 3, "I-LOC": 4,
        "B-ORG": 5, "I-ORG": 6,
        "B-TIME": 7, "I-TIME": 8
    }
}
