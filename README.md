# FreeLB++ AGNews Robustness Evaluation

This project trains a BERT classifier with a FreeLB++-style adversarial training loop and evaluates robustness on AGNews with TextFooler and TextBugger.

The implementation follows the paper setup for AGNews:

- train set: 20000 shuffled samples
- clean test set: 7000 shuffled samples
- attack evaluation set: 1000 samples selected from the shuffled clean test set
- attacks: TextFooler and TextBugger through TextAttack
- AGNews attack constraints: max modified word ratio 0.3, max synonym candidates 50, query budget defaulting to `50 * max_length`
- attack logs: `outputs/<attack>-<timestamp>.txt`

## Install

```powershell
pip install -r requirements.txt
```

TextAttack may download extra resources for attacks, such as NLTK data, counter-fitted embeddings, and sentence encoders.

## CPU Smoke Test

Use tiny sample sizes first to verify training on CPU:

```powershell
python src/train_freelbpp_agnews.py `
  --device cpu `
  --train-size 32 `
  --test-size 32 `
  --epochs 1 `
  --batch-size 4 `
  --adv-steps 1 `
  --max-length 64
```

Then verify one attack with a tiny attack set:

```powershell
python src/attack_agnews.py `
  --device cpu `
  --test-size 32 `
  --attack-size 4 `
  --max-length 64 `
  --attack textfooler
```

TextAttack resources can take time to download the first time.

## Full T4 Run

Train with the requested full AGNews sizes:

```powershell
python src/train_freelbpp_agnews.py `
  --device cuda `
  --train-size 20000 `
  --test-size 7000 `
  --epochs 3 `
  --batch-size 16 `
  --adv-steps 30 `
  --adv-lr 0.1 `
  --max-length 128
```

After training, run both attacks on the 1000-sample attack set:

```powershell
python src/attack_agnews.py `
  --device cuda `
  --test-size 7000 `
  --attack-size 1000 `
  --attack both
```

## Useful Options

- `--device cpu|cuda`: select CPU or GPU.
- `src/train_freelbpp_agnews.py`: train and optionally run clean evaluation.
- `src/attack_agnews.py`: load a checkpoint and run TextFooler/TextBugger.
- `--attack textfooler|textbugger|both`: choose the attack.
- `--checkpoint-dir`: where the model is saved or loaded from.
- `--output-dir`: where attack txt files are written.
- `--seed`: controls deterministic shuffling.
- `--adv-steps`: FreeLB++ ascent steps. The paper reports AGNews robustness peaking around 30 steps.
