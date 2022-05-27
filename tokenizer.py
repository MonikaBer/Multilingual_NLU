from transformers import BertTokenizer


class Tokenizer():
    def __init__(self, token_type):
        self.instance = None # must have this variable

        if(token_type == 'm-bert'):
            self.instance = BertTokenizer.from_pretrained(
                "bert-base-multilingual-cased",
                do_lower_case = False
            )

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)