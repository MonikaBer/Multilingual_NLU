import os
import torch
from torch import nn
from tqdm import tqdm
import transformers

from metrics import f1_score_func
from utils import *
from transformers import BertForQuestionAnswering

from pathlib import Path


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.num_labels = -1

    def set_optimizer(self, config):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = config.lr,
            eps = config.eps
        )

    def set_scheduler(self, config, num_steps):
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = config.warmup_steps,
            num_training_steps = num_steps * config.epochs
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['scheduler']
        del state['optimizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_checkpoint(self, epoch, config):
        Path(config.model_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f'{config.model_path}{self.__class__.__name__}_epoch_{epoch}.model')

    def load(self, epoch, config):
        Path(config.model_path).mkdir(parents=True, exist_ok=True)
        self.load_state_dict(torch.load(f'{config.model_path}{self.__class__.__name__}_epoch_{epoch}.model'))

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        #print('input_ids', input_ids.size())
        #print('attention_mask', attention_mask.size())
        #print('labels', labels.size())
        #exit()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return output



class RelationClassifier(BaseModel, nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()

        self.model = self._get_model(num_labels, config.device)
        self.num_labels = num_labels

    def _get_model(self, num_labels, device):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False
        )
        return model.to(device)
        
class EntityTagging(BaseModel, nn.Module):
    def __init__(self, config, num_labels, output_layer_size, loss_f):
        super().__init__()
        # 'bert-large-uncased-whole-word-masking-finetuned-squad'
        # 'bert-base-multilingual-cased'
        # 'bert-base-cased'
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-cased').to(config.device)
        self.num_labels = num_labels
        self.loss_f = loss_f

        self.relu = torch.nn.ReLU()

        self.linear_e1_s = torch.nn.Linear(output_layer_size, 512).to(config.device)
        self.linear_e1_e = torch.nn.Linear(output_layer_size, 512).to(config.device)
        self.linear_e2_s = torch.nn.Linear(output_layer_size, 512).to(config.device)
        self.linear_e2_e = torch.nn.Linear(output_layer_size, 512).to(config.device)

        self.linear2_e1_s = torch.nn.Linear(512, 1024).to(config.device)
        self.linear2_e1_e = torch.nn.Linear(512, 1024).to(config.device)
        self.linear2_e2_s = torch.nn.Linear(512, 1024).to(config.device)
        self.linear2_e2_e = torch.nn.Linear(512, 1024).to(config.device)

        self.linear3_e1_s = torch.nn.Linear(1024, output_layer_size).to(config.device)
        self.linear3_e1_e = torch.nn.Linear(1024, output_layer_size).to(config.device)
        self.linear3_e2_s = torch.nn.Linear(1024, output_layer_size).to(config.device)
        self.linear3_e2_e = torch.nn.Linear(1024, output_layer_size).to(config.device)

    def __getstate__(self):
        state = super().__getstate__()
        del state['loss_f']
        return state

    def default_forward(self, input_ids, attention_mask, exact_pos_in_token):
        #print('input_ids', input_ids.size())
        #print('input_ids', input_ids)
        #print('attention_mask', attention_mask.size())
        #print('start_positions', start_positions.size())
        #print('start_positions', start_positions)
        #print('end_positions', end_positions.size())
        #print('end_positions', end_positions)
        #exit()
        o_start, o_end = self.model( # ignore loss [4, 256]
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            #start_positions=start_positions, 
            #end_positions=end_positions,
            return_dict=False
        )
        
        os1 = self.linear_e1_s(self.relu(o_start))
        os2 = self.linear_e2_s(self.relu(o_start))
        oe1 = self.linear_e1_e(self.relu(o_end))
        oe2 = self.linear_e2_e(self.relu(o_end))

        os1 = self.linear2_e1_s(self.relu(os1))
        os2 = self.linear2_e2_s(self.relu(os2))
        oe1 = self.linear2_e1_e(self.relu(oe1))
        oe2 = self.linear2_e2_e(self.relu(oe2))
        
        os1 = self.linear3_e1_s(self.relu(os1))
        os2 = self.linear3_e2_s(self.relu(os2))
        oe1 = self.linear3_e1_e(self.relu(oe1))
        oe2 = self.linear3_e2_e(self.relu(oe2))

        output = torch.stack([os1, oe1, os2, oe2], dim=0) # <indices, batch, tokens number> [4, 3, 256]
        loss = self.loss_f(output, exact_pos_in_token)
        return loss, output

    def forward(self, model2_update_input_ids, model2_update_attention_mask, model2_update_labels, **kwargs): # other args ignore
        return self.default_forward(
            input_ids = model2_update_input_ids, 
            attention_mask = model2_update_attention_mask, 
            exact_pos_in_token = model2_update_labels
        )