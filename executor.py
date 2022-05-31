import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
from metrics import f1_score_func, accuracy_per_class, accuracy_per_class_QA

import dataloader
from dataset import DataSeqClassification, QADataset
from loss import QALossFunction, QAVectorLossFunction
from tokenizer import SpecialTokens

class Executor():
    def train_loop_QA(config, model, dataloader_train, dataloader_val, batch_processing: SpecialTokens):
        for epoch in tqdm(range(1, config.epochs + 1)):
            model.train()

            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc = 'Epoch {:1d}'.format(epoch), leave = False, disable = False)

            for batch_idx, batch in enumerate(progress_bar):
                if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
                    break
                model.zero_grad()
                batch = batch_processing.update_batch(
                    batch=batch, 
                    labels_to_add=batch['text_relation_labels'],
                    config=config,
                )

                loss, logits = model(**batch)

                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

                model.optimizer.step()
                model.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


            model.save_checkpoint(epoch, config)

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            # validation
            val_loss, predictions, true_vals = Executor.evaluate_QA(dataloader_val, model, config, batch_processing)
            #print('predictions', predictions)
            #print('true_vals', true_vals)
            val_f1 = f1_score_func_QA(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def test_QA(config, tokenizer, dataframe_test, model, batch_processing):
        tqdm.write('--------------------------------------------------------------------------------------')
        tqdm.write('##### TESTING #####')
        tqdm.write('--------------------------------------------------------------------------------------')
        # model.load_state_dict(torch.load(f'{config.model_path}_epoch_1.model', map_location = torch.device(config.device)))

        for it in dataframe_test.iter_df(): # because it is a generator, tuple does not work here
            df, lang, label_to_id, id_to_label = it[0], it[1], it[2], it[3]
            if (model.num_labels != len(label_to_id)):
                raise Exception(f"Wrong size of labels. For test labels must" +
                    f"match the size of the model labels which it was trained.\n" +
                    f"Model labels {model.num_labels}\nUsed labels now: {len(label_to_id)}\n" +
                    f"Labels now: {label_to_id}")

            dataset_test = QADataset(
                df=df, 
                max_length=config.max_length, 
                tokenizer=tokenizer,
                config=config,
                mode='test'
            )
            dataloader_test = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_test, 'test')

            tqdm.write(f'#### Test model for lang {lang} ####')

            _, predictions, true_vals = Executor.evaluate_QA(dataloader_test.dataloader, model, config, batch_processing)
            accuracy_per_class_QA(predictions, true_vals, id_to_label)

    def evaluate_QA(dataloader, model, config, batch_processing):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
                    break

                batch = batch_processing.update_batch(
                    batch=batch, 
                    labels_to_add=batch['text_relation_labels'],
                    config=config,
                )
                
                loss, logits = model(**batch)

                loss_val_total += loss.item()

                # to jest źle, trzeba zachować 4 wierzchołki
                #logits = torch.flatten(logits, end_dim=1)
                #target = torch.flatten(batch['exact_pos_in_token'])

                logits = logits.detach().cpu().numpy()
                label_ids = target.cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader)

        predictions = np.concatenate(predictions, axis = 0)
        true_vals = np.concatenate(true_vals, axis = 0)

        return loss_val_avg, predictions, true_vals


    def train_loop_relation(config, model, dataloader_train, dataloader_val):
        for epoch in tqdm(range(1, config.epochs + 1)):
            model.train()

            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc = 'Epoch {:1d}'.format(epoch), leave = False, disable = False)

            for batch_idx, batch in enumerate(progress_bar):
                if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
                    break
                model.zero_grad()

                loss, logits = model(**batch)
                
                loss_train_total += loss.item()
                loss.backward()

                # free memory
                #del inputs
                #del batchDevice
                #torch.cuda.empty_cache()

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

                model.optimizer.step()
                model.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


            model.save_checkpoint(epoch, config)

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            # validation
            val_loss, predictions, true_vals = Executor.evaluate(dataloader_val, model, config)
            val_f1 = f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def test(config, tokenizer, dataframe_test, model):
        tqdm.write('--------------------------------------------------------------------------------------')
        tqdm.write('##### TESTING #####')
        tqdm.write('--------------------------------------------------------------------------------------')
        # model.load_state_dict(torch.load(f'{config.model_path}_epoch_1.model', map_location = torch.device(config.device)))

        for it in dataframe_test.iter_df(): # because it is a generator, tuple does not work here
            df, lang, label_to_id = it[0], it[1], it[2]
            if (model.num_labels != len(label_to_id)):
                raise Exception(f"Wrong size of labels. For test labels must" +
                    f"match the size of the model labels which it was trained.\n" +
                    f"Model labels {model.num_labels}\nUsed labels now: {len(label_to_id)}\n" +
                    f"Labels now: {label_to_id}")

            dataset_test = DataSeqClassification(
                df=df, 
                max_length=config.max_length, 
                tokenizer=tokenizer,
                config=config,
                mode='test'
            )
            dataloader_test = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_test, 'test')

            tqdm.write(f'#### Test model for lang {lang} ####')

            _, predictions, true_vals = Executor.evaluate(dataloader_test.dataloader, model, config)
            accuracy_per_class(predictions, true_vals, label_to_id)

    def evaluate(dataloader, model, config):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
                    break
                
                loss, logits = model(**batch)

                loss_val_total += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = batch['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)
                
                #print('logits', logits)
                #print('true_vals', true_vals)
                #print(inputs['input_ids'])
                #print(inputs['attention_mask'])
                #print(true_vals)
                #exit(0)

        loss_val_avg = loss_val_total / len(dataloader)

        predictions = np.concatenate(predictions, axis = 0)
        true_vals = np.concatenate(true_vals, axis = 0)

        return loss_val_avg, predictions, true_vals

    

    def test_m1_m2(dataloader, model_1, model_2, config):
        tqdm.write('--------------------------------------------------------------------------------------')
        tqdm.write('##### TESTING #####')
        tqdm.write('--------------------------------------------------------------------------------------')
        # model.load_state_dict(torch.load(f'{config.model_path}_epoch_1.model', map_location = torch.device(config.device)))

        for it in dataframe_test.iter_df(): # because it is a generator, tuple does not work here
            df, lang, label_to_id = it[0], it[1], it[2]
            if (model.num_labels != len(label_to_id)):
                raise Exception(f"Wrong size of labels. For test labels must" +
                    f"match the size of the model labels which it was trained.\n" +
                    f"Model labels {model.num_labels}\nUsed labels now: {len(label_to_id)}\n" +
                    f"Labels now: {label_to_id}")

            dataset_test = DataSeqClassification(
                df=df, 
                max_length=config.max_length, 
                tokenizer=tokenizer,
                config=config,
                mode='test'
            )
            dataloader_test = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_test, 'test')

            tqdm.write(f'#### Test model for lang {lang} ####')

            _, predictions, true_vals = Executor.evaluate_m1_m2(dataloader_test.dataloader, model, config)
            accuracy_per_class(predictions, true_vals, label_to_id)

    def evaluate_m1_m2(dataloader, model, config):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if (config.fast_dev_run and batch_idx >= config.batch_fast_dev_run):
                    break
                
                loss, logits = model(**batch)

                loss_val_total += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = batch['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)
                
                #print('logits', logits)
                #print('true_vals', true_vals)
                #print(inputs['input_ids'])
                #print(inputs['attention_mask'])
                #print(true_vals)
                #exit(0)

        loss_val_avg = loss_val_total / len(dataloader)

        predictions = np.concatenate(predictions, axis = 0)
        true_vals = np.concatenate(true_vals, axis = 0)

        return loss_val_avg, predictions, true_vals