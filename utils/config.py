# -*- coding: utf-8 -*-
"""Config class"""

import json
from configs.config import CFG

class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data, train):
        self.data = data
        self.train = train

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)

#==========Testing the function============
if __name__ == '__main__':
    config = Config.from_json(CFG)
    #Data config
    print("Class names: ",config.data.class_names)
    print("Train: ",config.data.train_path)
    print("Test: ",config.data.test_path)
    print("Image_size: ",config.data.image_size)

    #Train config

    print("Batch_size: ",config.train.batch_size)
    print("Val_size: ",config.train.val_size)
    print("Val_size: ",config.train.buffer_size)