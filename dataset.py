import torch
from torch.utils.data import Dataset
import utils
from typing import Union
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re
import utils

class BaseDataFrame():
    def __init__(self):
        pass

    def load_data(self, dataset_path):
        df = pd.read_csv(dataset_path, sep = '\t')
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

    def _encode_labels(self, df):
        possible_labels = df.relation.unique()
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        return label_dict

    def remove_tags_from_df(self, df):
        for n, sentence in enumerate(df.text.values):
            df.text.values[n] = re.sub(re.compile('<.*?>'), '', sentence)
        print(df.text.values[2])
        return df


class TrainingDataFrame(BaseDataFrame):
    def __init__(self):
        self.path = None
        self.df = None
        self.encoded_labels = None

class TestDataFrame(BaseDataFrame):
    pass


class ProcessedDataFrame(TrainingDataFrame):
    def __init__(self, config):
        super().__init__()
        self.path = self.prepare_data(config)
        self.df, self.encoded_labels = self._prepare_df(config, self.path)

    def prepare_data(self, config):
        #Remove rare Relations from training and testing data
        for lang in config.langs:
            trainLangPath = config.data_dir + lang + '_corpora_train'
            testLangPath = config.data_dir + lang + '_corpora_test'
            trainLangDataset = pd.read_csv(trainLangPath + '.tsv', sep = '\t')
            testLangDataset = pd.read_csv(testLangPath + '.tsv', sep = '\t')
            train2LangDataset, test2LangDataset = \
                utils.remove_rare_relations_from_language_pair(trainLangDataset, testLangDataset)
            train2LangDataset.to_csv(trainLangPath + '2' + '.tsv', sep = '\t', index = False)
            test2LangDataset.to_csv(testLangPath + '2' + '.tsv', sep = '\t', index = False)

        # define path for joint train dataset
        dataset_path = config.data_dir
        if len(config.langs) > 1:
            dataset_path += 'NEW_'
        for lang in config.langs:
            dataset_path += lang + '_'
        dataset_path += "corpora_train2.tsv"

        # create joint dataset if it isn't exist
        if not os.path.exists(dataset_path):
            _create_joint_dataset(config.data_dir, config.langs, dataset_path)
        return dataset_path

    def _create_joint_dataset(data_dir, languages, new_dataset_path):
        with open(new_dataset_path, 'w') as fWrite:
            for lang_nr, lang in enumerate(languages):
                path = data_dir + lang + "_corpora_train2.tsv"
                with open(path) as fRead:
                    for line in fRead:
                        if lang_nr > 0 and line[:2] == "id":
                            continue
                        fWrite.write(line)

    def _prepare_df(self, config, dataset_path):
        print(f"Loading {dataset_path}")
        df = self.load_data(dataset_path)
        #print(possible_labels)
        encoded_labels = self._encode_labels(df)
        #print(encoded_labels)

        df['label'] = df.relation.replace(encoded_labels)
        #print(df.relation.value_counts())
        #print(df.index.values)
        df = self.remove_tags_from_df(df)

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

class ProcessedTestDataFrame(TestDataFrame):
    def __init__(self, config):
        self.langs = config.langs
        self.data_dir = config.data_dir

    def iter_df(self):
        for l in self.langs:
            test_dataset_path = self.data_dir + l + "_corpora_test2.tsv"
            test_df = self.load_data(test_dataset_path)
            encoded_labels = self._encode_labels(test_df)
            test_df['label'] = test_df.relation.replace(encoded_labels)
            yield test_df, l, encoded_labels


class DataSequence(Dataset):
    def __init__(self, df):
        super().__init__()
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

class DataSeqClassification(Dataset):
    def __init__(self, df: pd.DataFrame, max_length, tokenizer, config, mode: str):
        super().__init__()

        if(not isinstance(df, pd.DataFrame)):
            raise Exception(f"df not an instance of DataFrame: {type(df)}")

        if(mode == 'val'):
            df_in_use = df[df.data_type == 'val']
        elif(mode == 'train'):
            df_in_use = df[df.data_type == 'train']
        elif mode == 'test':
            df_in_use = df
        else:
            raise Exception("Unknown dataset type")

        txt = df_in_use.text.values.tolist()
        self.device = config.device

        self.encoded_data = tokenizer(
            txt,
            add_special_tokens = True,
            return_attention_mask = True,
            padding='max_length',
            max_length = max_length,
            return_tensors = 'pt'
        )

        # ids of the tokens in a sequence. Contains special reserved tokens
        self.input_ids = self.encoded_data['input_ids']
        # identify whether a token is a real token or padding
        self.attention_mask = self.encoded_data['attention_mask']
        self.labels = torch.tensor(df_in_use.label.values.tolist())

    def __len__(self):
        return len(self.labels)

    def get_attention_mask(self, idx):
        return self.attention_mask[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_input_ids(self, idx):
        return self.input_ids[idx]

    def __getitem__(self, idx):
        attention_mask = self.get_attention_mask(idx).to(self.device)
        label = self.get_label(idx).to(self.device)
        ids = self.get_input_ids(idx).to(self.device)

        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'labels': label
            }
