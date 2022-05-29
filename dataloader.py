import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from metrics import f1_score_func, accuracy_per_class
from utils import *


class BaseLoader():
    def __init__(self):
        pass

class SequenceClassificationDataLoader(BaseLoader):
    def __init__(self, config, tokenizer, dataset, dataloader_type):
        super().__init__()
        self.dataloader = self.create_loaders(config, tokenizer, dataset, dataloader_type)

    def create_loaders(self, config, tokenizer, dataset, dataloader_type):
        dataloader = self._get_dataloader(
            dataset=dataset,
            batch_size=config.batch_size,
            dataloader_type=dataloader_type,
        )
        return dataloader

    def _get_dataloader(self, dataset, batch_size, dataloader_type):
        if (dataloader_type == 'train'):
            sampler = RandomSampler(dataset)
        elif(dataloader_type == 'val'):
            sampler = SequentialSampler(dataset)
        elif(dataloader_type == 'test'):
            sampler = SequentialSampler(dataset)
        else:
            raise Exception("Unknown dataloader type")

        return DataLoader(
            dataset,
            sampler = sampler,
            batch_size = batch_size
        )


