from transformers import BertTokenizerFast


class Tokenizer():
    def __init__(self, token_type):
        self.instance = None # must have this variable

        if(token_type == 'm-bert'):
            self.instance = BertTokenizerFast.from_pretrained(
                "bert-base-multilingual-cased",
                do_lower_case = False
            )

        #self.instance.add_special_tokens({
        #    'additional_special_tokens': 
        #})

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)