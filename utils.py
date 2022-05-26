import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset

def load_data(file_path):
    df = pd.read_csv(file_path, sep = '\t')
    # remove the unnecessary columns
    columns = ['id', 'entity_1', 'entity_2', 'lang']
    toDropColumns = []
    for c in columns:
        if c in df:
            toDropColumns.append(c)
    if (len(toDropColumns) != 0):
        df = df.drop(columns = toDropColumns)

    # rename columns
    toRename = {'label':'relation'}
    toRenameConfirm = {}
    for key, val in toRename.items():
        if key in df:
            toRenameConfirm[key] = val
    df.rename(columns = toRenameConfirm, inplace = True)
    #print(df.head())
    return df

def encode_labels(possible_labels):
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    return label_dict

def prepare_df(dataset_path, config):
    print(f"Loading {dataset_path}")
    df = load_data(dataset_path)
    possible_labels = df.relation.unique()
    #print(possible_labels)
    encoded_labels = encode_labels(possible_labels)
    #print(encoded_labels)

    df['label'] = df.relation.replace(encoded_labels)
    #print(df.relation.value_counts())
    #print(df.index.values)

    # split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        df.index.values,
        df.label.values,
        test_size = config.test_size,
        random_state = config.random_state,
        stratify = df.label.values
    )

    df['data_type'] = ['not_set'] * df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    df.groupby(['relation', 'label', 'data_type']).count()

    return df, encoded_labels

def get_dataloader(tokenizer, df, max_length, batch_size, dataloader_type):
    encoded_data = tokenizer.batch_encode_plus(
        df.text.values,
        add_special_tokens = True,
        return_attention_mask = True,
        pad_to_max_length = True,
        max_length = max_length,
        return_tensors = 'pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(df.label.values)
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if dataloader_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler = sampler,
        batch_size = batch_size
    )


def get_model(encoded_labels, device):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels = len(encoded_labels),
        output_attentions = False,
        output_hidden_states = False
    )
    return model.to(device)

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader, model, device, config):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch_idx, batch in enumerate(dataloader):
        if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
            break
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader)

    predictions = np.concatenate(predictions, axis = 0)
    true_vals = np.concatenate(true_vals, axis = 0)

    return loss_val_avg, predictions, true_vals

def create_joint_dataset(data_dir, languages, new_dataset_path):
    with open(new_dataset_path, 'w') as fWrite:
        for lang_nr, lang in enumerate(languages):
            path = data_dir + lang + "_corpora_train2.tsv"
            with open(path) as fRead:
                for line in fRead:
                    if lang_nr > 0 and line[:2] == "id":
                        continue
                    fWrite.write(line)

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