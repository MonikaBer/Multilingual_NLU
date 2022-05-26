import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from metrics import f1_score_func, accuracy_per_class
from utils import *


class BaseLoader():
    def __init__(self):
        self.dataset_path = None

    def prepare_data(self, config):
        #Remove rare Relations from training and testing data
        for lang in config.langs:
            trainLangPath = config.data_dir + lang + '_corpora_train'
            testLangPath = config.data_dir + lang + '_corpora_test'
            trainLangDataset = pd.read_csv(trainLangPath + '.tsv', sep = '\t')
            testLangDataset = pd.read_csv(testLangPath + '.tsv', sep = '\t')
            train2LangDataset, test2LangDataset = remove_rare_relations_from_language_pair(trainLangDataset, testLangDataset)
            train2LangDataset.to_csv(trainLangPath + '2' + '.tsv', sep = '\t', index = False)
            test2LangDataset.to_csv(testLangPath + '2' + '.tsv', sep = '\t', index = False)

        # define path for joint train dataset
        self.dataset_path = config.data_dir
        if len(config.langs) > 1:
            self.dataset_path += 'NEW_'
        for lang in config.langs:
            self.dataset_path += lang + '_'
        self.dataset_path += "corpora_train2.tsv"

        # create joint dataset if it isn't exist
        if not os.path.exists(self.dataset_path):
            _create_joint_dataset(config.data_dir, config.langs, self.dataset_path)

    def _create_joint_dataset(data_dir, languages, new_dataset_path):
        with open(new_dataset_path, 'w') as fWrite:
            for lang_nr, lang in enumerate(languages):
                path = data_dir + lang + "_corpora_train2.tsv"
                with open(path) as fRead:
                    for line in fRead:
                        if lang_nr > 0 and line[:2] == "id":
                            continue
                        fWrite.write(line)

class SequenceClassificationDataLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased",
            do_lower_case = False
        )
        self.encoded_labels = None
        self.dataloader_train = None
        self.dataloader_val = None

    def create_loaders(self, config):
        df, self.encoded_labels = self._prepare_df(config)

        self.dataloader_train = self._get_dataloader(
            self.tokenizer,
            df[df.data_type == 'train'],
            config.max_length,
            config.batch_size,
            'train'
        )

        self.dataloader_val = self._get_dataloader(
            self.tokenizer,
            df[df.data_type == 'val'],
            config.max_length,
            config.batch_size,
            'val'
        )

    def _prepare_df(self, config):
        print(f"Loading {self.dataset_path}")
        df = self.load_data(self.dataset_path)
        possible_labels = df.relation.unique()
        #print(possible_labels)
        encoded_labels = self._encode_labels(possible_labels)
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

    def _encode_labels(self, possible_labels):
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        return label_dict

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
    
    def _get_dataloader(self, tokenizer, df, max_length, batch_size, dataloader_type):
        encoded_data = tokenizer.batch_encode_plus(
            df.text.values,
            add_special_tokens = True,
            return_attention_mask = True,
            padding='max_length',
            max_length = max_length,
            return_tensors = 'pt'
        )

        input_ids = encoded_data['input_ids']
        attention_mask = encoded_data['attention_mask']
        label = torch.tensor(df.label.values)
        dataset = TensorDataset(input_ids, attention_mask, label)

        if dataloader_type == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            sampler = sampler,
            batch_size = batch_size
        )


