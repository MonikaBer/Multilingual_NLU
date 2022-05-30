import os
import torch
from torch import nn
from tqdm import tqdm
import transformers

from metrics import f1_score_func
from utils import *
from transformers import BertForQuestionAnswering


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.num_labels = -1

    def set_optimizer(self, config):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = config.lr,
            eps = config.eps
        )

    def set_scheduler(self, config, num_steps):
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = config.warmup_steps,
            num_training_steps = num_steps * config.epochs
        )

    def save_checkpoint(self, epoch, config):
        torch.save(self.model.state_dict(), f'{config.model_path}_{self.__class__.__name__}_epoch_{epoch}.model')

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
        # 'bert-base-cased'
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(config.device)
        self.num_labels = num_labels
        self.loss_f = loss_f

        self.linear1 = torch.nn.Linear(output_layer_size, 256).to(config.device)
        self.linear2 = torch.nn.Linear(output_layer_size, 256).to(config.device)
        self.linear3 = torch.nn.Linear(output_layer_size, 256).to(config.device)
        self.linear4 = torch.nn.Linear(output_layer_size, 256).to(config.device)

    def forward(self, input_ids, attention_mask, exact_pos_in_token, **kwargs): # other args ignore
        #print('input_ids', input_ids.size())
        #print('input_ids', input_ids)
        #print('attention_mask', attention_mask.size())
        #print('start_positions', start_positions.size())
        #print('start_positions', start_positions)
        #print('end_positions', end_positions.size())
        #print('end_positions', end_positions)
        #exit()
        _, o = self.model( # ignore loss [4, 256]
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            #start_positions=start_positions, 
            #end_positions=end_positions,
            return_dict=False
        )
        o1 = self.linear1(o)
        o2 = self.linear2(o)
        o3 = self.linear3(o)
        o4 = self.linear4(o)
        output = torch.stack([o1, o2, o3, o4], dim=0) # <indices, batch, tokens number> [4, 4, 1024]
        loss = self.loss_f(output, exact_pos_in_token)
        return loss, output