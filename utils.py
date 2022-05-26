import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset

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

def remove_rare_relations(trainRelationDict, testRelationDict,
        trainUniqueRelationList, testUniqueRelationList):
    keysToRemove = []
    for k in trainRelationDict.keys():
        if (trainRelationDict[k] < 16):
            keysToRemove.append(k)
    if keysToRemove:
        for k in keysToRemove:
            if k in trainUniqueRelationList:
                trainUniqueRelationList.remove(k)
            if k in testUniqueRelationList:
                testUniqueRelationList.remove(k)
            del trainRelationDict[k]
            del testRelationDict[k]


def remove_rare_relations_from_language_pair(train, test):
    trainUniqueRelationList = train.label.unique().tolist()
    testUniqueRelationList = test.label.unique().tolist()
    trainUniqueRelationList.sort()
    testUniqueRelationList.sort()
    if trainUniqueRelationList != testUniqueRelationList:
        commonRelation = []
        for i in trainUniqueRelationList:
            for j in testUniqueRelationList:
                if i == j:
                    commonRelation.append(i)
        remove_uncommon_values_in_two_lists(trainUniqueRelationList, commonRelation)
        remove_uncommon_values_in_two_lists(testUniqueRelationList, commonRelation)
    trainRelationDict = count_relations(train, trainUniqueRelationList)
    testRelationDict = count_relations(test, testUniqueRelationList)
    remove_rare_relations(trainRelationDict, testRelationDict,
        trainUniqueRelationList, testUniqueRelationList)
    train = train[train.label.isin(trainUniqueRelationList)]
    test = test[test.label.isin(testUniqueRelationList)]
    return train, test

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

def align_label_example(tokenized_input, labels):
        word_ids = tokenized_input.word_ids()
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
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
      
        return label_ids








