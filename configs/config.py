# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "class_names": ['cat', 'dog'],
        "train_path": "data/dogs-vs-cats/train",
        "test_path": "data/dogs-vs-cats/test",
        "image_size": 128,
    },
    "train": {
        "batch_size": 32,
        "buffer_size": 1000,
        "val_size" : 0.2,
        "epoches": 10,
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-3,
        },
    },
}