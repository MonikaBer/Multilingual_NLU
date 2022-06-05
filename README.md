# Multilingual_NLU
Multilingual Natural Language Understanding with M-BERT and SMiLER dataset for Multilingual Entity and Relation Extraction.

## Doc
- [Initial documentation](https://demo.hedgedoc.org/0ezHN-JjQGm7Oog9j-Ty0A)
- [Final documentation](https://demo.hedgedoc.org/DvFln3INS12i5sH8q8tIfw)
- [Comments](https://demo.hedgedoc.org/T4G22XgsSHGTEUNk7J5s_w)

# Requirements
- Python 3.10.4
- Pip 22.0.4

## Configuration
- unpack archive with SMiLER data
```
cd data
tar -xzvf datasets.tar.gz
cd ..
```

- create virtual environment
```
python -m venv venv
source venv/bin/activate
```

## Execution
1. Main script for training execution:
```
python main.py --device "cuda" --max-length 256 --batch-size 32 --langs "(ru,pl,es)" --epochs 4
```

2. Script for generating experiments:
```
./scripts/exps_generator.sh --id=0 --results-path="example-experiments.csv"
```

3. Scripts for executing experiments:
```
./scripts/exps_executor.sh --hyperparams-path="example-experiments.csv" --start=0 --n=5 --r=3
```
