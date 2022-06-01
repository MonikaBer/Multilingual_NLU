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
import math

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

    def _encode_shortcut_label(self, df):
        possible_labels = df.label.unique()
        label_to_shortcut = {}
        shortcut_to_label = {}

        ascii_char = ord('0')

        for index, possible_label in enumerate(sorted(possible_labels)):
            label_to_shortcut[possible_label] = chr(ascii_char)
            shortcut_to_label[chr(ascii_char)] = possible_label
            ascii_char += 1
        
        return label_to_shortcut, shortcut_to_label

    def convert_to_ner_data(self, df: pd.DataFrame, tokenizer, config):
        texts = df.text.values.tolist()
        process = EntityFinding()

        def pr_labels(row):
            return process.new_target(
                text=row.text, 
                entity_1=row.entity_1, 
                entity_2=row.entity_2,
                tokenizer=tokenizer,
                config=config
            )

        def pr_text(row):
            return EntityFinding.new_text(row.text)

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

    def remove_invalid_data(self, df):
        return df[df.label_ner != 'None_wrong_record']

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
    def __init__(self, config, tokenizer, debug_path=None):
        if(debug_path is not None):
            self.path = debug_path
        else:
            self.path = self.prepare_data(config)
        self.df, self.label_to_id, self.id_to_label = self._prepare_df(config, self.path, tokenizer=tokenizer)

    def __len__(self):
        return self.df.shape[0]

    def _prepare_df(self, config, dataset_path, tokenizer):
        print(f"Loading {dataset_path}")
        df = self.load_data(dataset_path)
        new_df = self.convert_to_ner_data(df, tokenizer, config)

        label_to_id, id_to_label = self._encode_labels(new_df)
        new_df['label_id'] = new_df.apply(lambda row: label_to_id[row['label']], axis=1)

        label_to_shortcut, _ = self._encode_shortcut_label(new_df)
        new_df['label_shortcut'] = new_df.apply(lambda row: label_to_shortcut[row['label']], axis=1)

        # split dataset
        new_df = self.split_data(new_df, config)
        new_df = self.remove_invalid_data(new_df)

        return new_df, label_to_id, id_to_label

class ProcessedTestDataFrame(TestDataFrame):
    def __init__(self, config, tokenizer):
        self.config = config
        self.langs = config.langs
        self.data_dir = config.data_dir
        self.tokenizer = tokenizer

    def iter_df(self):
        dfs = []
        for l in self.langs:
            test_dataset_path = self.data_dir + l + "_corpora_test2.tsv"
            test_df = self.load_data(test_dataset_path)
            dfs.append(test_df)

        full_df = pd.concat(dfs, ignore_index=True)
        label_to_id, id_to_label = self._encode_labels(full_df) # from full dataframe to be unique
        label_to_shortcut, _ = self._encode_shortcut_label(full_df)

        for test_df, l in zip(dfs, self.langs):
            test_df['label_id'] = test_df.apply(lambda row: label_to_id[row['label']], axis=1)
            test_df['label_shortcut'] = test_df.apply(lambda row: label_to_shortcut[row['label']], axis=1)
            test_df = self.convert_to_ner_data(test_df, self.tokenizer, self.config)
            test_df = self.remove_invalid_data(test_df)
            yield test_df, l, label_to_id, id_to_label, label_to_shortcut



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
        self.regex_gen_number = re.compile('[IB]\-ent(.*)')

        if(mode == 'val'):
            df_in_use = df[df.data_type == 'val']
        elif(mode == 'train'):
            df_in_use = df[df.data_type == 'train']
        elif mode == 'test':
            df_in_use = df
        else:
            raise Exception("Unknown dataset type")

        self.txt = df_in_use.text_ner.values.tolist()
        self.device = config.device
        self.labels = df_in_use.label_ner.values
        self.label_id = torch.tensor(df_in_use.label_id.values.tolist())
        self.text_relation_labels = df_in_use.label_shortcut.values.tolist()

        self.encoded_data = tokenizer(
            self.txt,
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

        if(len(self.labels) != len(self.input_ids)):
            raise Exception("Bad number of rows after tokenization.")

        # wygląda ok
        #print(txt[1])
        #print(self.labels[1])
        #print(self.get_label(1))
        #exit()
        self.tokenizer = tokenizer

    def convert_ner_label_to_indices(self, label: list):
        """
            Returns list of 4 indices.
        """
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

    def get_ids_size(self):
        return len(self.input_ids[0])

class QADataset(TaggingDataset):
    def __init__(self, df: pd.DataFrame, max_length, tokenizer, config, mode: str):
        super().__init__(df, max_length, tokenizer, config, mode)

    def convert_QA_label_to_indices(self, label: list):
        e1_end = None
        e2_end = None

        start_idxs = [-1, -1]
        end_idxs = [-1, -1]
        for idx, l in enumerate(label):
            if l == self.e1_name_start or l == self.e2_name_start:
                number = int(self.regex_gen_number.findall(l)[0]) - 1
                start_idxs[number] = idx
            elif l == self.e1_name_end or l == self.e2_name_end:
                number = int(self.regex_gen_number.findall(l)[0]) - 1
                end_idxs[number] = idx
        #print('aaa', start_idxs, end_idxs)
        #print('label', label)
        idx = 0
        if(end_idxs[idx] == -1):
            end_idxs[idx] = start_idxs[idx]
        idx = 1
        if(end_idxs[idx] == -1):
            end_idxs[idx] = start_idxs[idx]
        #print('bbb', start_idxs, end_idxs)    
        return start_idxs, end_idxs
        
    def get_label_classification(self, idx, ids_size):
        # returns tensor like [0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0]
        lab = self.labels[idx]
        default = [0.0] * ids_size # for example to 256, not to len(lab) = 35
        for idx, l in enumerate(lab):
            result = self.regex_gen_number.findall(l)
            if(len(result) != 0):
                default[idx] = float(result[0])
        return default

    def convert_to_tokenized_word(self, indexes_start: list, indexes_end: list, idx):
        '''
            Converts ids from word count to token count.
        '''
        word_ids = self.encoded_data.word_ids(idx)
        """ # not needed if use EntityFinding
        counted_max_words = max([i for i in word_ids if i is not None])
        text_split_count = len(utils.split_text(self.txt[idx]))
        if(counted_max_words != text_split_count):
            raise Exception("Wrong word split count. The number of words splited by this alg. is "+
            "different than the number of words counted by tokenizer."+
            f"\nword_ids count: {counted_max_words}\nText split count: {text_split_count}\nIdx: {idx}"+
            f"\nTokenized word: {self.tokenizer.instance.decode(self.input_ids[idx])}" +
            f"\nword_ids: {word_ids}\nText: {self.txt[idx]}")
        """

        new_indexes_start = [-1, -1]
        new_indexes_end = [-1, -1]
        # assuming word_ids is increasing
        print('ccc', indexes_start, indexes_end)
        print(word_ids)
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            # early stop
            if(w_id == indexes_start[0] and new_indexes_start[0] == -1):
                new_indexes_start[0] = idx
            elif(w_id == indexes_start[1] and new_indexes_start[1] == -1):
                new_indexes_start[1] = idx

            # do not stop
            if(w_id == indexes_end[0]):
                new_indexes_end[0] = idx
            elif(w_id == indexes_end[1]):
                new_indexes_end[1] = idx
        print('ggg', new_indexes_start, new_indexes_end)

        return new_indexes_start, new_indexes_end

    def convert_ner_to_labels_indices(self, idx):
        start_positions, end_positions = self.convert_QA_label_to_indices(self.labels[idx])

        if(-1 in end_positions or -1 in start_positions):
            raise Exception(f"Could not find start or end for row {idx}. Check data csv for errors.\n" +
            f"start: {start_positions}\nend: {end_positions}\ntext: {self.txt[idx]}\nLabel: {self.labels[idx]}")

        exact_pos_in_token = torch.tensor([
            start_positions[0],
            end_positions[0],
            start_positions[1],
            end_positions[1]
        ]).to(self.device)

        #print(exact_pos_in_token)
        #print(self.df['text'][idx])
        #print(self.df['text_ner'][idx])
        #print(self.tokenizer.instance.convert_ids_to_tokens(self.input_ids[idx]))
        #exit()
        return exact_pos_in_token

    def __getitem__(self, idx):
        '''
            Returns labels in form of 4 indices <e1_start, e1_end, e2_start, e2_end>
        '''
        attention_mask = self.get_attention_mask(idx).to(self.device)
        ids = self.get_input_ids(idx).to(self.device)
        processed_text = self.txt[idx]

        exact_pos_in_token = self.convert_ner_to_labels_indices(idx)

        #vector_label = torch.tensor(self.get_label_classification(idx, len(ids))).to(self.device)
        #start_positions, end_positions = self.convert_QA_label_to_indices(self.labels[idx])
        ##print(self.tokenizer.instance.convert_ids_to_tokens(self.input_ids[idx]))
        ##start_positions, end_positions = self.convert_to_tokenized_word(start_positions, end_positions, idx)
        #exact_pos_in_token = torch.tensor([
        #    start_positions[0],
        #    end_positions[0],
        #    start_positions[1],
        #    end_positions[1]
        #]).to(self.device)
        #start_positions = torch.tensor(start_positions, dtype=torch.long).to(self.device)
        #end_positions = torch.tensor(end_positions, dtype=torch.long).to(self.device)


        
        #default_labels = self.get_label(idx).to(self.device)

        

        #start_positions = start_positions[0]
        #end_positions = end_positions[0]

        #if(-1 in end_positions or -1 in start_positions):
        #    raise Exception(f"Could not find start or end for row {idx}. Check data csv for errors.\n" +
        #    f"start: {start_positions}\nend: {end_positions}\ntext: {self.txt[idx]}")

        #print(vector_label.size())
        #print(exact_pos_in_token.size())
        #print(ids.size())

        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            #'start_positions': start_positions,
            #'end_positions': end_positions,
            'exact_pos_in_token': exact_pos_in_token,
            #'vector_label': vector_label,
            #'labels': default_labels,
            'labels': self.label_id[idx].to(self.device),
            'text_relation_labels': self.text_relation_labels[idx],
            'text': processed_text
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
        '''
            Converts from data EntityContainer (indexes of the tags <eX>) to form
            ['B-ent1', 'I-ent1', '0', '0', '0', '0', '0', '0', 'B-ent2', '0', '0', '0', '0', '0', '0', '0']
        '''
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

# should not be used, old impl
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
        """
            Converts from text string with tags <eX>, </eX> to target labels
            to format like ['B-ent1', 'I-ent1', '0', '0', '0', '0', '0', '0', 'B-ent2', '0', '0', '0', '0', '0', '0', '0']
        """
        sum_numb_of_removed_words = 0
        buffer = EntityContainer()
        sum_numb_of_added_words = 0

        splits = utils.split_text(text)
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



class SublistFinder():
    def __init__(self, sublist: list):
        new_sublist = []
        stop_pad = False
        for s in sublist:
            if not (s is None or s == '[CLS]' or s == '[PAD]'):
                new_sublist.append(s)

        self.sublist = new_sublist
        self.pos = 0
        self.at_start_pos = -1
        self.at_last_pos = -1

    def process(self, token: str, at_pos):
        #print(token, "== sublist ", self.sublist, "== at_pos ", at_pos, "== pos ", self.pos, "==", len(self.sublist))
        if(self.pos == len(self.sublist)):
            # found and it is OK
            return
        if(self.sublist[self.pos].lower() == token.lower()):
            if(self.pos == 0):
                self.at_start_pos = at_pos
            self.pos += 1
            self.at_last_pos = at_pos
        else:
            self.pos = 0

    def found_sublist(self) -> bool:
        #print(self.sublist, "==", self.pos, "==", len(self.sublist))
        return self.pos == len(self.sublist)

    def get_start_pos(self):
        return self.at_start_pos

    def get_end_pos(self):
        return self.at_last_pos

class EntityFinding():
    def __init__(self):
        self.none_value = 'None_wrong_record'

    def new_text(text: str):
        new_text = re.sub('<e[0-9]+>|<\/e[0-9]+>', ' ', text) # add default space, two spaces do nothing wrong
        return new_text    

    def new_target(self, text: str, entity_1, entity_2, tokenizer, config):
        '''
            Find entities in text by text comparsion with entity_1 and entity_2.
            Convert to format like 
            ['B-ent1', 'I-ent1', '0', '0', '0', '0', '0', '0', 'B-ent2', '0', '0', '0', '0', '0', '0', '0']
            but this format is relative to the tokenized word
        '''
        if((isinstance(entity_1, float) and math.isnan(entity_1)) or 
            (isinstance(entity_2, float) and math.isnan(entity_2))):
            return self.none_value
        text = EntityFinding.new_text(text)

        encoded_text = tokenizer(
            text,
            add_special_tokens = True,
            return_attention_mask = True,
            padding='max_length',
            max_length = config.max_length,
            return_tensors = 'pt'
        )

        tokenized_text = tokenizer.instance.convert_ids_to_tokens(
            encoded_text.input_ids[0]
        )
        tokenized_text = list(filter(lambda a: a != '[PAD]' and a != '[CLS]', tokenized_text))

        text_word_ids = encoded_text.word_ids()

        tokenized_entity_1 = tokenizer.instance.convert_ids_to_tokens(
            tokenizer(
                entity_1,
                add_special_tokens = True,
                return_attention_mask = True,
                padding='max_length',
                max_length = config.max_length,
                return_tensors = 'pt'
            ).input_ids[0]
        )
        tokenized_entity_1 = list(filter(lambda a: a != '[PAD]' and a != '[CLS]', tokenized_entity_1))
        if tokenized_entity_1[-1] == '[SEP]':
            tokenized_entity_1.pop()
        
        tokenized_entity_2 = tokenizer.instance.convert_ids_to_tokens(
            tokenizer(
                entity_2,
                add_special_tokens = True,
                return_attention_mask = True,
                padding='max_length',
                max_length = config.max_length,
                return_tensors = 'pt'
            ).input_ids[0]
        )
        tokenized_entity_2 = list(filter(lambda a: a != '[PAD]' and a != '[CLS]', tokenized_entity_2))
        if tokenized_entity_2[-1] == '[SEP]':
            tokenized_entity_2.pop()

        if(tokenized_entity_1 is None or tokenized_entity_2 is None):
            return self.none_value

        e1_finder = SublistFinder(tokenized_entity_1)
        e2_finder = SublistFinder(tokenized_entity_2)

        entity_found = [False, False]
        for idx, token in enumerate(tokenized_text):
            e1_finder.process(token, idx)
            e2_finder.process(token, idx)

            if(e1_finder.found_sublist()):
                e1_start = e1_finder.get_start_pos()
                e1_end = e1_finder.get_end_pos()
                entity_found[0] = True
            if(e2_finder.found_sublist()):
                e2_start = e2_finder.get_start_pos()
                e2_end = e2_finder.get_end_pos()
                entity_found[1] = True
            if(entity_found[0] and entity_found[1]):
                break

        
        if not (entity_found[0] and entity_found[1]):
            #print(entity_found)
            return self.none_value

        #print(tokenized_text)
        #print(e1_start)
        #print(e1_end)
        #print(e2_start)
        #print(e2_end)
        target = self._convertToTarget(e1_start, e1_end, e2_start, e2_end, config.max_length)
        #print(text)
        #print(target)
        return target
        
    def _convertToTarget(self, e1_start, e1_end, e2_start, e2_end, numb_tokens):
        '''
            Converts from data EntityContainer (indexes of the tags <eX>) to form
            ['B-ent1', 'I-ent1', '0', '0', '0', '0', '0', '0', 'B-ent2', '0', '0', '0', '0', '0', '0', '0']
        '''

        special_one_line_idxs = []
        # first iteration - set start and end
        target = ['0'] * numb_tokens

        target[e1_end] = 'I-ent1'
        target[e2_end] = 'I-ent2'
        target[e1_start] = 'B-ent1'
        target[e2_start] = 'B-ent2'

        #print(target)

        # second iteration - fill empty space between start and end
        repeat = False
        to_repeat = None
        for idx, t in enumerate(reversed(target)):
            idx = len(target) - idx - 1
            if 'I-ent' in t:
                repeat = True
                to_repeat = t
            elif 'B-ent' in t:
                repeat = False

            if(repeat):
                target[idx] = to_repeat
        #print(target)

        ok = [False, False]
        for idx, t in enumerate(target):
            if 'B-ent1' in t:
                ok[0] = True
            if 'B-ent2' in t:
                ok[1] = True
        if(ok[0] == False or ok[1] == False):
            return self.none_value

        return target

        


    