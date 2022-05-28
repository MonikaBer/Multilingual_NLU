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

def align_label(texts, tokenizer):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
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

def which_tag(tag):
    if tag[1] == 'e':
        if tag[2] == '##1':
            return 0
        elif tag[2] == '##2':
            return 2
        else:
            return -1
    elif tag[1] == '/':
        if tag[3] == '##1':
            return 1
        elif tag[3] == '##2':
            return 3
        else:
            return -1
    else:
        return -1


def find_entity_token_positions(df):
    '''
    df must contain sentences with <e1>, <e2> tags
    '''

    #tags for <e1>, </e1>, <e2>, </e2> are always returned this way
    b1 = ['<', 'e', '##1', '>']
    e1 = ['<', '/', 'e', '##1', '>']
    b2 = ['<', 'e', '##2', '>']
    e2 = ['<', '/', 'e', '##2', '>']

    tagListLong = [b1,e1,b2,e2]
    tagList = [b1,e1[0:4],b2,e2[0:4]]

    positionFourEntitiesArray = []
    possibleErrorList = []
    for sentenceNumber, sentence in enumerate(df.text.values):
        #values used in next loop must be initialized
        positionFourEntities = []
        point = -4
        increment = -1
        for n,i in enumerate(sentence):
            if i == '<':
                point = n
                tagCandidate = []
            if n - point < 4:
                flag = 0
                for tag in tagList:
                    if tag[n - point] == i:
                        tagCandidate.append(i)
                        flag = 1
                        break
                if flag == 0:
                    point = n - 4
                if (len(tagCandidate) - 1 == n - point) and (flag == 1):
                    if n - point == 3:
                        tagNumber = which_tag(tagCandidate)
                        increment += 1
                        if tagNumber != increment:
                            possibleErrorList.append(sentenceNumber)
                            point = n - 4
                        else:
                            positionFourEntities.append(n - 3)
                            point = n - 4
            positionFourEntitiesArray.append(positionFourEntities)
    if possibleErrorList:
        warnings.warn('there are problems with entity tags or with entity_token_positions function')

    #return positions of first token in every entity (out of 4) for each sentence
    #dimensions [len(df.text.values) x 4]
    return torch.tensor(positionFourEntitiesArray)
