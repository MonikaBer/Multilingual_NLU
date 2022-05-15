import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


MAX_LENGTH = 8 #256
DEVICE = "cpu" #cuda


def load_data(file_path):
    df = pd.read_csv(file_path, sep = '\t')
    # remove the unnecessary columns
    df = df.drop(columns = ['id', 'entity_1', 'entity_2', 'lang'])
    df.rename(columns = {'label':'relation'}, inplace = True)
    #print(df.head())
    return df

def encode_labels(possible_labels):
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    return label_dict


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels, encoded_labels):
    label_dict_inverse = {v: k for k, v in encoded_labels.items()}

    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader_val, model):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(DEVICE) for b in batch)

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

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis = 0)
    true_vals = np.concatenate(true_vals, axis = 0)

    return loss_val_avg, predictions, true_vals


def main():
    parser = ArgumentParser()
    args = parser.parse_args()

    # 1. load data
    df = load_data('data/datasets/ru_corpora_train.tsv')
    possible_labels = df.relation.unique()
    # print(possible_labels)

    encoded_labels = encode_labels(possible_labels)
    #print(encoded_labels)

    df['label'] = df.relation.replace(encoded_labels)
    #print(df.relation.value_counts())

    #print(df.index.values)


    # 2. split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        df.index.values,
        df.label.values,
        test_size = 0.15,
        random_state = 42,
        stratify = df.label.values
    )

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    df.groupby(['relation', 'label', 'data_type']).count()

    #print(df)


    # 3. preprocess data
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].text.values,
        add_special_tokens = True,
        return_attention_mask = True,
        pad_to_max_length = True,
        max_length = MAX_LENGTH,
        return_tensors = 'pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].text.values,
        add_special_tokens = True,
        return_attention_mask = True,
        pad_to_max_length = True,
        max_length = MAX_LENGTH,
        return_tensors = 'pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


    # 4. build model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels = len(encoded_labels),
        output_attentions = False,
        output_hidden_states = False
    )
    model.to(DEVICE)


    # 5. dataloaders
    batch_size = 3

    dataloader_train = DataLoader(
        dataset_train,
        sampler = RandomSampler(dataset_train),
        batch_size = batch_size
    )

    dataloader_validation = DataLoader(
        dataset_val,
        sampler = SequentialSampler(dataset_val),
        batch_size = batch_size
    )


    # 6. optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr = 1e-5,
        eps = 1e-8
    )

    epochs = 1

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = len(dataloader_train) * epochs
    )


    # 7. training loop
    seed_val = 17
    set_seed(seed_val)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc = 'Epoch {:1d}'.format(epoch), leave = False, disable = False)

        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(DEVICE) for b in batch)

            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


        torch.save(model.state_dict(), f'models/finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')


    # 8. validation
    model.load_state_dict(torch.load('models/finetuned_BERT_epoch_1.model', map_location = torch.device(DEVICE)))

    _, predictions, true_vals = evaluate(dataloader_validation, model)
    accuracy_per_class(predictions, true_vals, encoded_labels)


    return 0


if __name__ == "__main__":
    exit(main())
