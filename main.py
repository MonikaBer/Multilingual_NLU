# import pdb
import os
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


def evaluate(dataloader_val, model, device):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

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

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis = 0)
    true_vals = np.concatenate(true_vals, axis = 0)

    return loss_val_avg, predictions, true_vals


'''
arg:
  string - ex. "(ru, es, en, ...)"
returns:
  tuple - ex. ("ru", "es", "en", ...)
'''
def str2list(s):
    s = s.replace('(', '').replace(')', '').replace(' ', '')
    list_str = map(str, s.split(','))
    return list(list_str)


def create_joint_dataset(data_dir, languages, new_dataset_path):
    with open(new_dataset_path, 'w') as fWrite:
        for lang_nr, lang in enumerate(languages):
            path = data_dir + lang + "_corpora_train.tsv"
            with open(path) as fRead:
                for line in fRead:
                    if lang_nr > 0 and line[:2] == "id":
                        continue
                    fWrite.write(line)


def main():
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--data-dir", type = str, default = "data/datasets/",
                        help = "path to directory with datasets (default: %(default)s)")
    parser.add_argument("--langs", type = str, default = "(ru,fa)",
                        help = "tuple of languages (default: %(default)s)")
    parser.add_argument("--model-path", type = str, default = "models/model1",
                        help = "model path for storage (default: %(default)s)")
    # device dependencies
    parser.add_argument("--device", type = str, default = "cuda",
                        help = "device ex. cuda, cpu (default: %(default)s)")
    parser.add_argument("--batch-size", type = int, default = 32,
                        help = "batch size (default: %(default)s)")
    parser.add_argument("--max-length", type = int, default = 256,
                        help = "max length (default: %(default)s)")
    # another experiment parameters
    parser.add_argument("--epochs", type = int, default = 4,
                        help = "number of epochs (default: %(default)s)")
    parser.add_argument("--random-state", type = int, default = 42,
                        help = "random state (default: %(default)s)")
    parser.add_argument("--test-size", type = float, default = 0.15,
                        help = "size of test set (default: %(default)s)")
    parser.add_argument("--lr", type = float, default = 1e-5,
                        help = "learning rate (default: %(default)s)")
    parser.add_argument("--eps", type = float, default = 1e-8,
                        help = "epsilon (default: %(default)s)")
    parser.add_argument("--warmup-steps", type = int, default = 0,
                        help = "number of warmup steps (default: %(default)s)")
    parser.add_argument("--seed", type = int, default = 17,
                        help = "seed (default: %(default)s)")
    parser.add_argument("--max-norm", type = float, default = 1.0,
                        help = "max norm of the gradients (default: %(default)s)")
    args = parser.parse_args()

    # 1. load data
    languages = str2list(args.langs)
    languages.sort()
    print(languages)

    # define path for joint dataset
    new_dataset_path = args.data_dir
    if len(languages) > 1:
        new_dataset_path += 'NEW_'
    for lang in languages:
        new_dataset_path += lang + '_'
    new_dataset_path += "corpora_train.tsv"

    # create joint dataset if it isn't exist
    if not os.path.exists(new_dataset_path):
        create_joint_dataset(args.data_dir, languages, new_dataset_path)


    df = load_data(new_dataset_path)
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
        test_size = args.test_size,
        random_state = args.random_state,
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
        max_length = args.max_length,
        return_tensors = 'pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].text.values,
        add_special_tokens = True,
        return_attention_mask = True,
        pad_to_max_length = True,
        max_length = args.max_length,
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
    model.to(args.device)


    # 5. dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        sampler = RandomSampler(dataset_train),
        batch_size = args.batch_size
    )

    dataloader_validation = DataLoader(
        dataset_val,
        sampler = SequentialSampler(dataset_val),
        batch_size = args.batch_size
    )


    # 6. optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        eps = args.eps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = len(dataloader_train) * args.epochs
    )


    # 7. training loop
    set_seed(args.seed)

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc = 'Epoch {:1d}'.format(epoch), leave = False, disable = False)

        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(args.device) for b in batch)

            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})


        torch.save(model.state_dict(), f'{args.model_path}_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_validation, model, args.device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')


    # 8. validation
    model.load_state_dict(torch.load(f'{args.model_path}_epoch_1.model', map_location = torch.device(args.device)))

    _, predictions, true_vals = evaluate(dataloader_validation, model, args.device)
    accuracy_per_class(predictions, true_vals, encoded_labels)


    return 0


if __name__ == "__main__":
    exit(main())
