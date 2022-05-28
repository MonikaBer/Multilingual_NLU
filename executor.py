import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
from metrics import f1_score_func, accuracy_per_class

import dataloader
from dataset import DataSeqClassification

class Executor():
    def train_loop_tagging(model, df_train, df_val):

        train_dataset = DataSequence(df_train)
        val_dataset = DataSequence(df_val)

        train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

        if use_cuda:
            model = model.cuda()

        best_acc = 0
        best_loss = 1000

        for epoch_num in range(EPOCHS):

            total_acc_train = 0
            total_loss_train = 0

            model.train()

            for train_data, train_label in tqdm(train_dataloader):

                train_label = train_label[0].to(device)
                mask = train_data['attention_mask'][0].to(device)
                input_ids = train_data['input_ids'][0].to(device)

                optimizer.zero_grad()
                loss, logits = model(input_ids, mask, train_label)

                logits_clean = logits[0][train_label != -100]
                label_clean = train_label[train_label != -100]

                predictions = logits_clean.argmax(dim=1)

                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

                loss.backward()
                optimizer.step()

            model.eval()

            total_acc_val = 0
            total_loss_val = 0

            for val_data, val_label in val_dataloader:

                val_label = val_label[0].to(device)
                mask = val_data['attention_mask'][0].to(device)

                input_ids = val_data['input_ids'][0].to(device)

                loss, logits = model(input_ids, mask, val_label)

                logits_clean = logits[0][val_label != -100]
                label_clean = val_label[val_label != -100]

                predictions = logits_clean.argmax(dim=1)          

                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

            val_accuracy = total_acc_val / len(df_val)
            val_loss = total_loss_val / len(df_val)

            print(
                f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

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

                #print('loss', loss.size())
                #print('logits', logits.size())
                #exit(0)
                
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
            df, lang, encoded_labels = it[0], it[1], it[2]

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
            accuracy_per_class(predictions, true_vals, encoded_labels)

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
                #print(inputs['input_ids'])
                #print(inputs['attention_mask'])
                #print(true_vals)
                #exit(0)

        loss_val_avg = loss_val_total / len(dataloader)

        predictions = np.concatenate(predictions, axis = 0)
        true_vals = np.concatenate(true_vals, axis = 0)

        return loss_val_avg, predictions, true_vals
