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


class Config:
    def __init__(
        self,
        data_dir,
        langs,
        model_path,
        device,
        batch_size,
        max_length,
        epochs,
        random_state,
        test_size,
        lr,
        eps,
        warmup_steps,
        seed,
        max_norm,
        fast_dev_run,
        batch_fast_dev_run
    ):
        self.data_dir = data_dir
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs
        self.random_state = random_state
        self.test_size = test_size
        self.lr = lr
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.max_norm = max_norm
        self.fast_dev_run = fast_dev_run
        self.batch_fast_dev_run = batch_fast_dev_run

        languages = str2list(langs)
        languages.sort()
        print(languages)
        self.langs = languages
