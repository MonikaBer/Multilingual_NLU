from transformers import BertTokenizerFast
import torch

class Tokenizer():
    def __init__(self, token_type):
        self.instance = None # must have this variable

        if(token_type == 'm-bert'):
            self.instance = BertTokenizerFast.from_pretrained(
                "bert-base-multilingual-cased",
                do_lower_case = False
            )
        elif(token_type == 'large-bert'):
            self.instance = BertTokenizerFast.from_pretrained(
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                do_lower_case = False
            )

        #self.instance.add_special_tokens({
        #    'additional_special_tokens': 
        #})

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)


class SpecialTokens():
    def __init__(self, label_keys: list[str], tokenizer, model):
        new_label_keys = []
        label_to_id = {}
        for l in label_keys:
            new_label_keys.append('[' + l + ']')

        tokenizer.instance.add_special_tokens({"additional_special_tokens": new_label_keys})
        model.model.resize_token_embeddings(len(tokenizer.instance))
        for l in label_keys:
            label_to_id[l] = tokenizer('[' + l + ']')['input_ids'][1]

        self.label_to_id = label_to_id
        self.label_keys = new_label_keys
        self.sep_token_id = tokenizer('[SEP]')['input_ids'][1]
        self.tokenizer = tokenizer

    def update_batch(self, batch, labels_to_add: list[str], config):
        '''
            labels_to_add - must have the size of <batch>
        '''
        text = batch['text']

        model2_input_ids = []
        model2_attention_mask = []
        model2_labels = []
        for txt, token_pos, label_to_add in zip(batch['text'], batch['exact_pos_in_token'], labels_to_add):
            new_txt = label_to_add + '[SEP]' + txt
            tokenized_data = self.tokenizer(
                new_txt,
                add_special_tokens = True,
                return_attention_mask = True,
                padding='max_length',
                max_length = config.max_length,
                return_tensors = 'pt'
            )
            #print(batch['input_ids'][0])
            #print(self.tokenizer.instance.decode(tokenized_data['input_ids'][0]))
            #print(tokenized_data['input_ids'][0])
            #exit()
            model2_input_ids.append(tokenized_data['input_ids'][0])
            model2_attention_mask.append(tokenized_data['attention_mask'][0])
            model2_labels.append(token_pos.add(2)) # shift by two positions (added two tags to beginning)
            
                
        #print(type(model2_input_ids))
        #print(model2_input_ids)
        batch['model2_update_input_ids'] = torch.stack(model2_input_ids).to(config.device)
        batch['model2_update_attention_mask'] = torch.stack(model2_attention_mask).to(config.device)
        batch['model2_update_labels'] = torch.stack(model2_labels).to(config.device)

        # looks good
        #print(batch)
        
        return batch


    # not implemented, should not be used
    def update_batch_test(self, batch, label_to_add: str):
        '''
            This function changes input_ids, attention_mask, exact_pos_in_token in batch.
        '''

        for ids, mask, target in zip(batch['input_ids'], batch['attention_mask'], batch['exact_pos_in_token']):
            ids = ids.tolist()
            mask = mask.tolist()
            target = target.tolist()

            idx = len(mask)
            for m in reversed(mask):
                if m == 1:
                    mask[idx] = 1
                    mask[idx + 1] = 1
                    break
                idx -= 1

            target.pop(-1)
            target.pop(-1)
            target.insert(1, self.sep_token_id)
            target.insert(1, self.label_to_id[label_to_add])

            
            ids.pop(-1)
            ids.pop(-1)

            
            





