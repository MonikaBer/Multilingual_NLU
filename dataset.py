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
        #df = self.remove_tags_from_df(df)

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


# regex lookup for <eX> </eX>
class EntityDataClass():
    def __init__(self, entity_number=None, start=None, end=None):
        self.entity_number = entity_number
        self.start = start
        self.end = end

class EntityContainer():
    def __init__(self):
        self.entities = {}

    def add(self, entity_number, start=None, end=None):
        if not entity_number in self.entities:
            self.entities[entity_number] = EntityDataClass(
                entity_number=entity_number,
                start=start,
                end=end,
            )
        else:
            if start is not None:
                self.entities[entity_number].start = start
            if end is not None:
                self.entities[entity_number].end = end

    def __len__(self):
        return len(self.entities)

    def _convertToTarget(self, numb_entities, numb_words):
        if(numb_entities != len(self.entities)):
            raise Exception(f"Found different number of entities. Assumed number: {numb_entities}. Real number: {len(self.entities)}")


        special_one_line_idxs = []
        # first iteration - set start and end
        target = ['0'] * numb_words
        for entity_numb, v in self.entities.items():
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
        self.numb_entities = numb_entities

    def _getRealIndex(self, index, numb_found):
        return index - numb_found

    def process(self, text: str):
        numb_found = 0
        buffer = EntityContainer()

        splits = text.split()
        for idx, s in enumerate(splits):
            start = None
            end = None
            entity_number = None

            # s can be like <e1>abc</e1> !!
            lookahead = self.regex_lookahead.findall(s)
            if(len(lookahead) != 0):
                start = idx
                lookahead = self.regex_entity_start.findall(s)
                entity_number = self.regex_get_entity_number.findall(lookahead[0])

            lookback = self.regex_lookback.findall(s)
            if(len(lookback) != 0):
                end = idx
                lookback = self.regex_entity_end.findall(s)
                entity_number = self.regex_get_entity_number.findall(lookback[0])

            # if only a signle entity without other characters like '<e1>'
            if start is None and end is None:
                single = self.regex.findall(s)
                if(len(single) != 0):
                    print(single)
                    entity_number = self.regex_get_entity_number.findall(single[0])
                    if(len(self.regex_entity_start.findall(single)) != 0):
                        buffer.add(entity_number=entity_number[0], start=self._getRealIndex(idx, numb_found))
                    else:
                        buffer.add(entity_number=entity_number[0], end=self._getRealIndex(idx, numb_found))
                    numb_found += 1
            else:
                buffer.add(entity_number=entity_number[0], start=start, end=end)

        return buffer._convertToTarget(self.numb_entities, len(splits) - numb_found)

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
