from argparse import ArgumentParser

from config import Config
from RelationClassifier import RelationClassifier
from EntityTagger import EntityTagger

import utils


def main():
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
    parser.add_argument("--batch-fdr", type = int, default = 5,
                        help = "limit number of batches in each epoch (default: %(default)s)")
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

    if args.task == "R":
        model = RelationClassifier(config)
    else:
        raise NotImplementedError()
        model = EntityTagger(config)

    model.prepare_data()
    model.create_dataloaders()
    model.build_model()

    #size = utils.getModelSize(model.model)
    #print(size)
    #exit(0)
    model.set_optimizer()
    model.set_scheduler()

    model.train()
    model.test()

    return 0


if __name__ == "__main__":
    exit(main())
