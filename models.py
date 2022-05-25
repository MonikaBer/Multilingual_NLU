import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from metrics import f1_score_func, accuracy_per_class
from utils import *


class RelationClassifier:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased",
            do_lower_case = False
        )

    def prepare_data(self):
        #Remove rare Relations from training and testing data
        for lang in self.config.langs:
            trainLangPath = self.config.data_dir + lang + '_corpora_train'
            testLangPath = self.config.data_dir + lang + '_corpora_test'
            trainLangDataset = load_data(trainLangPath + '.tsv')
            testLangDataset = load_data(testLangPath + '.tsv')
            train2LangDataset, test2LangDataset = remove_rare_relations_from_language_pair(trainLangDataset, testLangDataset)
            train2LangDataset.to_csv(trainLangPath + '2' + '.tsv', sep = '\t', index = False)
            test2LangDataset.to_csv(testLangPath + '2' + '.tsv', sep = '\t', index = False)


        # define path for joint train dataset
        self.dataset_path = self.config.data_dir
        if len(self.config.langs) > 1:
            self.dataset_path += 'NEW_'
        for lang in self.config.langs:
            self.dataset_path += lang + '_'
        self.dataset_path += "corpora_train2.tsv"

        # create joint dataset if it isn't exist
        if not os.path.exists(self.dataset_path):
            create_joint_dataset(self.config.data_dir, self.config.langs, self.dataset_path)

    def create_dataloaders(self):
        df, self.encoded_labels = prepare_df(self.dataset_path, self.config)

        self.dataloader_train = get_dataloader(
            self.tokenizer,
            df[df.data_type == 'train'],
            self.config.max_length,
            self.config.batch_size,
            'train'
        )

        self.dataloader_val = get_dataloader(
            self.tokenizer,
            df[df.data_type == 'val'],
            self.config.max_length,
            self.config.batch_size,
            'val'
        )

    def build_model(self):
        self.model = get_model(self.encoded_labels, self.config.device)

    def set_optimizer(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.config.lr,
            eps = self.config.eps
        )

    def set_scheduler(self):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = self.config.warmup_steps,
            num_training_steps = len(self.dataloader_train) * self.config.epochs
        )

    def train(self):
        set_seed(self.config.seed)

        for epoch in tqdm(range(1, self.config.epochs + 1)):
            self.model.train()

            loss_train_total = 0

            progress_bar = tqdm(self.dataloader_train, desc = 'Epoch {:1d}'.format(epoch), leave = False, disable = False)

            for batch in progress_bar:
                self.model.zero_grad()

                batchDevice = tuple(b.to(self.config.device) for b in batch)

                inputs = {
                    'input_ids':      batchDevice[0],
                    'attention_mask': batchDevice[1],
                    'labels':         batchDevice[2],
                }

                outputs = self.model(**inputs)
                
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                # free memory
                del inputs
                del batchDevice
                torch.cuda.empty_cache()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)

                self.optimizer.step()
                self.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


            self.save_checkpoint(epoch)

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(self.dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            # validation
            val_loss, predictions, true_vals = evaluate(self.dataloader_val, self.model, self.config.device)
            val_f1 = f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f'{self.config.model_path}_epoch_{epoch}.model')

    def test(self):
        tqdm.write('--------------------------------------------------------------------------------------')
        tqdm.write('##### TESTING #####')
        tqdm.write('--------------------------------------------------------------------------------------')
        # self.model.load_state_dict(torch.load(f'{self.config.model_path}_epoch_1.model', map_location = torch.device(self.config.device)))

        for lang in self.config.langs:
            test_dataset_path = self.config.data_dir + lang + "_corpora_test2.tsv"
            test_df = load_data(test_dataset_path)
            test_df['label'] = test_df.relation.replace(self.encoded_labels)

            dataloader_test = get_dataloader(
                self.tokenizer,
                test_df,
                self.config.max_length,
                self.config.batch_size,
                'test'
            )

            tqdm.write(f'#### Test model for lang {lang} ####')

            _, predictions, true_vals = evaluate(dataloader_test, self.model, self.config.device)
            accuracy_per_class(predictions, true_vals, self.encoded_labels)
