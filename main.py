from argparse import ArgumentParser

from config import Config
from model import RelationClassifier, EntityTagging
from EntityTagger import EntityTagger

import torch
import utils
import dataloader
from  executor import Executor
from tokenizer import Tokenizer
from dataset import (
    DataSeqClassification,
    ProcessedTestDataFrame,
    ProcessTokens,
    TaggingDataset,
    TrainHERBERTaDataFrame,
    QADataset,
)

from loss import QAVectorLossFunction


def get_parser():
    parser = ArgumentParser()
    # task type
    parser.add_argument("--task", type = str, default = "R",
                        help = "task type ('R' - for relations classification, 'E' - for entities tagging) (default: %(default)s)")
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
    parser.add_argument("--fast-dev-run", action='store_true',
                        help = "option for development")
    parser.add_argument("--batch-fdr", type = int, default = 2,
                        help = "limit number of batches in each epoch (default: %(default)s)")
    return parser

def train_loop(config, my_data_frame, tokenizer, model):
    # przenoszę to tu aby python mógł przy po treningu i przy teście zwolnić trochę pamięci

    #
    # test
    #
    '''texts = my_data_frame.df.text.values.tolist()
    process = ProcessTokens()

    for idx, i in enumerate(texts):
        ret = process.process(i)
        print(i)
        print(ret)
        print('-------------------------------------------------------------')
        if(idx == 6):
            break
        
    exit(0)'''
    #
    # end test
    #

    dataset_val = DataSeqClassification(
        df=my_data_frame.df, 
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='val'
    )   
    dataset_train = DataSeqClassification(
        df=my_data_frame.df, 
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='train'
    )   

    dataloader_val = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_val, 'val')
    dataloader_train = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_train, 'train')
    
    model.set_scheduler(config, num_steps=len(dataloader_train.dataloader))
    
    #new_label = utils.align_label(
    #    texts=data.df.text.values.tolist(),
    #    tokenizer=tokenizer
    #)
    #print(new_label)
    #print(model.tokenizer.convert_ids_to_tokens(data.encoded_data["input_ids"][0]))
    #print((data.encoded_data["attention_mask"][0]))
    #exit()

    Executor.train_loop_relation(
        config=config, 
        model=model, 
        dataloader_train=dataloader_train.dataloader, 
        dataloader_val=dataloader_val.dataloader
    )

def test_loop(config, tokenizer, model):
    my_data_frame = ProcessedTestDataFrame(config)

    Executor.test(
        config=config,
        tokenizer=tokenizer,
        dataframe_test=my_data_frame,
        model=model
    )

def for_model_1(config):
    tokenizer = Tokenizer('m-bert')  

    my_data_frame = TrainHERBERTaDataFrame(config)

    ### test
    """datase = TaggingDataset(
        df=my_data_frame.df, 
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='val'
    )

    exit()
    """
    ### end test

    model = RelationClassifier(config, len(my_data_frame.label_to_id))
    model.set_optimizer(config)
    
    train_loop(
        config=config,
        model=model,
        my_data_frame=my_data_frame,
        tokenizer=tokenizer
    )

    test_loop(
        config=config,
        model=model,
        tokenizer=tokenizer
    )

def for_model_2(config):
    tokenizer = Tokenizer('large-bert')  
    my_data_frame = TrainHERBERTaDataFrame(config)
    
    dataset_val = QADataset(
        df=my_data_frame.df, 
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='val'
    )   
    dataset_train = QADataset(
        df=my_data_frame.df, 
        max_length=config.max_length, 
        tokenizer=tokenizer,
        config=config,
        mode='train'
    )   
    
    loss_f = QAVectorLossFunction(torch.nn.CrossEntropyLoss())
    model = EntityTagging(config, len(my_data_frame.label_to_id), dataset_train.get_ids_size(), loss_f=loss_f)
    model.set_optimizer(config)

    dataloader_val = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_val, 'val')
    dataloader_train = dataloader.SequenceClassificationDataLoader(config, tokenizer, dataset_train, 'train')
    
    model.set_scheduler(config, num_steps=len(dataloader_train.dataloader))

    Executor.train_loop_QA(
        config=config, 
        model=model, 
        dataloader_train=dataloader_train.dataloader, 
        dataloader_val=dataloader_val.dataloader)

    my_test_data_frame = ProcessedTestDataFrame(config)

    Executor.test_QA(
        config=config,
        tokenizer=tokenizer,
        dataframe_test=my_test_data_frame,
        model=model
    )

def main():
    parser = get_parser()
    args = parser.parse_args()

    config = Config(
        data_dir = args.data_dir,
        langs = args.langs,
        model_path = args.model_path,
        device = args.device,
        batch_size = args.batch_size,
        max_length = args.max_length,
        epochs = args.epochs,
        random_state = args.random_state,
        test_size = args.test_size,
        lr = args.lr,
        eps = args.eps,
        warmup_steps = args.warmup_steps,
        seed = args.seed,
        max_norm = args.max_norm,
        fast_dev_run = args.fast_dev_run,
        batch_fast_dev_run = args.batch_fdr,
    )
    utils.set_seed(config.seed)

    if args.task == "R":
        for_model_1(config)
    else:
        for_model_2(config)

    return 0


if __name__ == "__main__":
    exit(main())
