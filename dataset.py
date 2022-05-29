import torch
from torch.utils.data import Dataset
import utils
from typing import Union
import pandas as pd
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import re
import utils

class BaseDataFrame():
    def __init__(self):
        pass

    def _remove_columns(self, df):
        columns = ['id', 'entity_1', 'entity_2', 'lang']
        toDropColumns = []
        for c in columns:
            if c in df:
                toDropColumns.append(c)
        if (len(toDropColumns) != 0):
            df = df.drop(columns = toDropColumns)
        return df

    def _rename_columns(self, df):
        toRename = {'label':'relation'}
        toRenameConfirm = {}
        for key, val in toRename.items():
            if key in df:
                toRenameConfirm[key] = val
        df.rename(columns = toRenameConfirm, inplace = True)
        return df

    def load_data(self, dataset_path):
        df = pd.read_csv(dataset_path, sep = '\t')
        #df = self._remove_columns(df)
        #df = self._rename_columns(df)
        return df

    def _encode_labels(self, df):
        possible_labels = df.label.unique()
        label_to_id = {}
        id_to_label = {}

        for index, possible_label in enumerate(sorted(possible_labels)):
            label_to_id[possible_label] = index
            id_to_label[index] = possible_label

        return label_to_id, id_to_label

    def convert_to_ner_data(self, df: pd.DataFrame):
        texts = df.text.values.tolist()
        process = ProcessTokens()

        def pr_labels(row):
            return process.new_target(row.text)

        def pr_text(row):
            return ProcessTokens.new_text(row.text)

        df['label_ner'] = df.apply(pr_labels, axis=1)
        df['text_ner'] = df.apply(pr_text, axis=1)

        return df

    def split_data(self, df: pd.DataFrame, config):
        # return train, val, test
        train, val = np.split(df.sample(frac=1, random_state=42),
                            [int((1.0 - config.test_size) * len(df))]) # split as 0, 0.85, 1.0
        train['data_type'] = 'train'
        val['data_type'] = 'val'
        new_df = pd.concat([train, val])
        return new_df

class TrainingDataFrame(BaseDataFrame):
    def __init__(self):
        self.path = None
        self.df = None
        self.label_to_id = None

    @staticmethod
    def _create_joint_dataset(data_dir, languages, new_dataset_path):
        dfs = []
        for lang_nr, lang in enumerate(languages):
            path = data_dir + lang + "_corpora_train2.tsv"
            dfs.append(pd.read_csv(path, sep = '\t'))
        new_df = pd.concat(dfs, ignore_index=True)  # .drop_duplicates(subset='id').reset_index(drop=True) # if id can be duplicated
        new_df.to_csv(new_dataset_path, sep='\t', index=False)

    def prepare_data(self, config):
        #Remove rare Relations from training and testing data
        for lang in config.langs:
            trainLangPath = config.data_dir + lang + '_corpora_train'
            testLangPath = config.data_dir + lang + '_corpora_test'
            if(not os.path.isfile(trainLangPath + '2' + '.tsv') or not os.path.isfile(testLangPath + '2' + '.tsv')):
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
            self._create_joint_dataset(config.data_dir, config.langs, dataset_path)
        return dataset_path

class TestDataFrame(BaseDataFrame):
    pass



class TrainHERBERTaDataFrame(TrainingDataFrame):
    def __init__(self, config):
        self.path = self.prepare_data(config)
        self.df, self.label_to_id, self.id_to_label = self._prepare_df(config, self.path)

    def remove_invalid_data(self, df):
        return df[df.label != 'None_wrong_record']

    def _prepare_df(self, config, dataset_path):
        print(f"Loading {dataset_path}")
        df = self.load_data(dataset_path)
        new_df = self.convert_to_ner_data(df)

        label_to_id, id_to_label = self._encode_labels(new_df)
        new_df['label_id'] = new_df.apply(lambda row: label_to_id[row['label']], axis=1)

        # split dataset
        new_df = self.split_data(new_df, config)
        new_df = self.remove_invalid_data(new_df)

        return new_df, label_to_id, id_to_label

class ProcessedTestDataFrame(TestDataFrame):
    def __init__(self, config):
        self.langs = config.langs
        self.data_dir = config.data_dir

    def iter_df(self):
        dfs = []
        for l in self.langs:
            test_dataset_path = self.data_dir + l + "_corpora_test2.tsv"
            test_df = self.load_data(test_dataset_path)
            dfs.append(test_df)

        full_df = pd.concat(dfs, ignore_index=True)
        label_to_id, _ = self._encode_labels(full_df) # from full dataframe to be unique

        for test_df, l in zip(dfs, self.langs):
            test_df['label_id'] = test_df.apply(lambda row: label_to_id[row['label']], axis=1)
            test_df = self.convert_to_ner_data(test_df)
            yield test_df, l, label_to_id



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

        txt = df_in_use.text_ner.values.tolist()
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
        self.labels = torch.tensor(df_in_use.label_id.values.tolist())

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

        #print("-----------------------------")
        #print('ids', ids)
        #print('txt', self.txt[idx])
        #print('label', label)
        #exit()

        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'labels': label
            }

class TaggingDataset(Dataset):
    # the assumption is that there are only 2 entities

    def __init__(self, df: pd.DataFrame, max_length, tokenizer, config, mode: str):
        '''
            Parse data in format NER where labels are in format like
            ['B-ent1', 'I-ent1', '0', '0', '0', '0', '0', '0', 'B-ent2', '0', '0', '0', '0', '0', '0', '0']
        '''
        if(not isinstance(df, pd.DataFrame)):
            raise Exception(f"df not an instance of DataFrame: {type(df)}")

        self.e1_name_start = 'B-ent1'
        self.e1_name_end = 'I-ent1'
        self.e2_name_start = 'B-ent2'
        self.e2_name_end = 'I-ent2'

        if(mode == 'val'):
            df_in_use = df[df.data_type == 'val']
        elif(mode == 'train'):
            df_in_use = df[df.data_type == 'train']
        elif mode == 'test':
            df_in_use = df
        else:
            raise Exception("Unknown dataset type")

        txt = df_in_use.text_ner.values.tolist()
        self.device = config.device
        self.labels = df_in_use.label_ner.values

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

        # wygląda ok
        #print(txt[1])
        #print(self.labels[1])
        #print(self.get_label(1))
        #exit()

    def convert_ner_label_to_indices(self, label: list):
        e1_end = None
        e2_end = None
        for idx, l in enumerate(label):
            if l == self.e1_name_start:
                e1_start = idx
            elif l == self.e2_name_start:
                e2_start = idx
            elif l == self.e1_name_end:
                e1_end = idx
            elif l == self.e2_name_end:
                e2_end = idx
        if(e1_end is None):
            e1_end = e1_start
        if(e2_end is None):
            e2_end = e2_start

        return [e1_start, e1_end, e2_start, e2_end]

    def __len__(self):
        return len(self.input_ids)

    def get_attention_mask(self, idx):
        return self.attention_mask[idx]

    def get_label(self, idx):
        indices = self.convert_ner_label_to_indices(self.labels[idx])
        return torch.tensor(indices)

    def get_input_ids(self, idx):
        return self.input_ids[idx]

    def __getitem__(self, idx):
        '''
            Returns labels in form of 4 indices <e1_start, e1_end, e2_start, e2_end>
        '''
        attention_mask = self.get_attention_mask(idx).to(self.device)
        label = self.get_label(idx).to(self.device)
        ids = self.get_input_ids(idx).to(self.device)

        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'labels': label
            }




# regex lookup for <eX> </eX>
class EntityDataClass():
    def __init__(self, entity_number=None, start=None, end=None, numb_of_added_words=0):
        self.entity_number = entity_number
        self.start = start
        self.end = end
        self.numb_of_added_words = numb_of_added_words

class EntityContainer():
    def __init__(self):
        self.entities = {}
        self.none_value = 'None_wrong_record'

    def add(self, entity_number, start=None, end=None, numb_of_added_words=0):
        if not entity_number in self.entities:
            self.entities[entity_number] = EntityDataClass(
                entity_number=entity_number,
                start=start,
                end=end,
                numb_of_added_words=numb_of_added_words
            )
        else:
            if start is not None:
                self.entities[entity_number].start = start
            if end is not None:
                self.entities[entity_number].end = end
            if numb_of_added_words != 0:
                self.entities[entity_number].numb_of_added_words += numb_of_added_words

    def __len__(self):
        return len(self.entities)

    def _convertToTarget(self, numb_entities, numb_words):
        if(numb_entities != len(self.entities)):
            return self.none_value
            #raise Exception(f"Found different number of entities. Assumed number: {numb_entities}. Real number: {len(self.entities)}")

        special_one_line_idxs = []
        # first iteration - set start and end
        target = ['0'] * numb_words
        for entity_numb, v in self.entities.items():
            if(v.start == None or v.end == None): # special case of error in input data
                return self.none_value
            target[v.start] = f'B-ent{entity_numb}' # beginning
            if(v.start != v.end):
                target[v.end] = f'I-ent{entity_numb}' # continuation
            else:
                special_one_line_idxs.append(v.start)

        # second iteration - fill empty space between start and end
        repeat = False
        to_repeat = None
        for idx, t in enumerate(target):
            if 'B-ent' in t:
                repeat = True
                tmp = t.replace('B-ent', '')
                to_repeat = f'I-ent{tmp}'
                if idx in special_one_line_idxs:
                    repeat = False
                continue
            elif 'I-ent' in t:
                repeat = False
                continue

            if(repeat):
                target[idx] = to_repeat

        return target

class ProcessTokens():
    def __init__(self, numb_entities=2):
        self.regex = re.compile("<e[0-9]+>|<\/e[0-9]+>")
        self.regex_get_entity_number = re.compile("[0-9]+")
        self.regex_entity_start = re.compile("<e[0-9]+>")
        self.regex_entity_end = re.compile("<\/e[0-9]+>")
        self.regex_lookahead = re.compile('<e[0-9]+>(.*)')
        self.regex_lookback = re.compile('(.*)<\/e[0-9]+>')
        self.regex_prefix_str = re.compile('(.*)<e[0-9]+>')
        self.regex_sufix_str = re.compile('<\/e[0-9]+>(.*)')
        self.numb_entities = numb_entities

    def _getRealIndex(self, index, sum_numb_of_removed_words, sum_numb_of_added_words):
        return index - sum_numb_of_removed_words + sum_numb_of_added_words

    def new_text(text: str):
        new_text = re.sub('<e[0-9]+>|<\/e[0-9]+>', ' ', text) # add default space, two spaces do nothing wrong
        return new_text

    def new_target(self, text: str):
        sum_numb_of_removed_words = 0
        buffer = EntityContainer()
        sum_numb_of_added_words = 0

        splits = text.split()
        for idx, s in enumerate(splits):
            numb_of_added_words = 0 # if the one word need spaces, like '(<e1>abc</e1>,' need 2 spaces for (,
            start = None
            end = None
            entity_number = None

            # s can be like aaa<e1>abc</e1>aaa !!
            lookahead = self.regex_lookahead.findall(s)
            if(len(lookahead) != 0): # <e1>abc
                start = idx
                lookahead = self.regex_entity_start.findall(s)
                entity_number = self.regex_get_entity_number.findall(lookahead[0])
                if(len(self.regex_prefix_str.findall(s)) != 0): # abc<e1>abc
                        numb_of_added_words += 1
                        sum_numb_of_added_words += 1

            lookback = self.regex_lookback.findall(s)
            if(len(lookback) != 0): # abc</e1>
                end = idx
                lookback = self.regex_entity_end.findall(s)
                entity_number = self.regex_get_entity_number.findall(lookback[0])
                if(len(self.regex_sufix_str.findall(s)) != 0): # abc</e1>abc
                    numb_of_added_words += 1
                    sum_numb_of_added_words += 1

            # if only a signle entity without other characters like '<e1>'
            if start is None and end is None:
                single = self.regex.findall(s)
                if(len(single) != 0):
                    print(single)
                    entity_number = self.regex_get_entity_number.findall(single[0])
                    if(len(self.regex_entity_start.findall(single)) != 0):
                        buffer.add(
                            entity_number=entity_number[0],
                            start=self._getRealIndex(idx, sum_numb_of_removed_words, sum_numb_of_added_words),
                            numb_of_added_words=numb_of_added_words
                        )
                    else:
                        buffer.add(
                            entity_number=entity_number[0],
                            end=self._getRealIndex(idx, sum_numb_of_removed_words, sum_numb_of_added_words),
                            numb_of_added_words=numb_of_added_words
                        )
                    sum_numb_of_removed_words += 1
            elif (entity_number is not None):
                buffer.add(entity_number=entity_number[0], start=start, end=end, numb_of_added_words=numb_of_added_words)

        return buffer._convertToTarget(
            numb_entities=self.numb_entities,
            numb_words=len(splits) - sum_numb_of_removed_words + sum_numb_of_added_words
        )

    def processInput(self, encoded_data):
        tokens = encoded_data['input_ids']
        word_ids = tokens.word_ids()

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
