
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
    ProcessedTestDataFrame,
    ProcessTokens,
    TaggingDataset,
    TrainHERBERTaDataFrame,
    QADataset,
)

from loss import QAVectorLossFunction


def test1():
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

    tokenizer = Tokenizer('large-bert')  
    dataframe = TrainHERBERTaDataFrame(config, debug_path="data/datasets/NEW_es_pl_ru_corpora_train2.tsv")
    train = QADataset(
        df=dataframe.df,
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='train'
    )

    val = QADataset(
        df=dataframe.df,
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='val'
    )

    for idx, _ in enumerate(dataframe.df):
        get = val[idx]
    #print(get)

if __name__ == "__main__":
    test1()