
from argparse import ArgumentParser

from config import Config
from model import RelationClassifier, EntityTagging
from EntityTagger import EntityTagger

import torch
import utils
import dataloader
from  executor import Executor
from tokenizer import Tokenizer
from dataset import (
    DataSeqClassification,
    TaggingDataset,
    TrainHERBERTaDataFrame,
    QADataset,
)

from loss import QAVectorLossFunction


def test2():
    config = Config(
        data_dir = "data/datasets/",
        langs = '(ru,pl)',
        device = "cpu",
        batch_size = 2,
        max_length = 256,
        seed=17,
        test_size=0.5,
    )
    utils.set_seed(config.seed)

    tokenizer = Tokenizer('m-bert')

    dataframe = TrainHERBERTaDataFrame(config, tokenizer=tokenizer, debug_path="data/datasets/NEW_fa_ru_corpora_train2.tsv")
    label_keys = list(dataframe.label_to_id.keys())
    for idx, l in enumerate(label_keys):
        label_keys[idx] = '[' + l + ']'

    print(label_keys)

    #tokenizer.instance.add_tokens(label_keys)
    #label_str = ' '.join(label_keys)

    tokenizer.instance.add_special_tokens({"additional_special_tokens": label_keys})
    print(tokenizer.instance.SPECIAL_TOKENS_ATTRIBUTES)
    print(tokenizer.instance.tokenize('[SEP]'))
    #tokenizer.instance.ge
    print(tokenizer('[SEP]', max_length=256, padding='max_length', return_tensors = 'pt'))
    print('[unused0]' in tokenizer.instance.vocab)


