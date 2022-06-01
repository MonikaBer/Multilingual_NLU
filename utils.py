import torch
import random
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from transformers.tokenization_utils_base import BatchEncoding
from tokenizer import Tokenizer

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def remove_uncommon_values_in_two_lists(list1, list2):
    #list 1 contains additional elemnts
    elementsToRemove = []
    for i in list1:
        flag = 1
        for j in list2:
            if i == j:
                flag = 0
        if flag == 1:
            elementsToRemove.append(i)
    for i in elementsToRemove:
        list1.remove(i)

def count_relations(dataset, datasetUniqueRelationList):
    datasetRelationDict = dict.fromkeys(datasetUniqueRelationList, 0)
    datasetRelationList = dataset["label"].tolist()
    for i in datasetRelationList:
        for j in datasetUniqueRelationList:
            if i == j:
                datasetRelationDict[i] += 1
    return datasetRelationDict

def relations_to_multiply(trainRelationDict,
        trainUniqueRelationList, relationThreshold):
        keysToMultiply = []
        for k in trainRelationDict.keys():
            if (trainRelationDict[k] < relationThreshold):
                keysToMultiply.append(k)
        return keysToMultiply

def append_rare_relations_to_df(train, keysToMultiply, relationThreshold):
    if relationThreshold and keysToMultiply:
        toMultiply_df = pd.DataFrame(columns = list(train.columns.values))
        for rel in keysToMultiply:
            rareRelationIndices = []
            for n,i in enumerate(train.text):
                if train.iloc[n]["label"] == rel:
                    rareRelationIndices.append(n)
            for m in range(relationThreshold - len(rareRelationIndices)):
                    toMultiply_df.loc[m] = train.loc[rareRelationIndices[m]]
            train = pd.concat([train, toMultiply_df], ignore_index = True)
    return train


def getModelSize(model):
    """
        Get model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def align_label(texts, labels, tokenizer):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()
    print(word_ids)
    raise Exception()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def split_text(txt):
    return txt.split()
