import os
import torch
from torch import nn
from tqdm import tqdm
import transformers

from metrics import f1_score_func, accuracy_per_class
from utils import *


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        self.model = None

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

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return output



class RelationClassifier(BaseModel, nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()

        self.model = self._get_model(num_labels, config.device)

    def _get_model(self, num_labels, device):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False
        )
        return model.to(device)
        
        

class EntityTagging(BaseModel):
    def __init__(self):
        super().__init__(config)

class BertModel(torch.nn.Module):
    def __init__(self, config, encoded_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(encoded_labels)).to(config.device)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return output

class BertModelSpecial(torch.nn.Module):
    def __init__(self, config, encoded_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(encoded_labels * 4)).to(config.device)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return output