import argparse
import json
from datetime import datetime
from pathlib import Path

from common import (
    get_device,
    load_agnews_splits,
    load_checkpoint_model_and_tokenizer,
    set_seed,
    summarize_attack_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FreeLB++ AGNews BERT with TextAttack.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/freelbpp-agnews-bert")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--test-size", type=int, default=7000)
    parser.add_argument("--attack-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--attack", choices=["textfooler", "textbugger", "both"], default="both")
    parser.add_argument("--max-synonyms", type=int, default=50)
    parser.add_argument("--max-modification-rate", type=float, default=0.3)
    parser.add_argument("--semantic-similarity", type=float, default=0.84)
    parser.add_argument(
        "--query-budget",
        type=int,
        default=None,
        help="TextAttack uses one global query budget. Default is max_synonyms * max_length.",
    )
    parser.add_argument("--attack-batch-size", type=int, default=32)
    parser.add_argument("--verbose-attack", action="store_true")
    return parser.parse_args()


def build_attack(attack_name: str, model_wrapper, max_modification_rate: float):
    from textattack.attack_recipes import TextBuggerLi2018, TextFoolerJin2019

    if attack_name == "textfooler":
        attack = TextFoolerJin2019.build(model_wrapper)
    elif attack_name == "textbugger":
        attack = TextBuggerLi2018.build(model_wrapper)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    try:
        from textattack.constraints.overlap import MaxWordsPerturbed

        attack.constraints = [c for c in attack.constraints if not isinstance(c, MaxWordsPerturbed)]
        attack.constraints.append(MaxWordsPerturbed(max_percent=max_modification_rate))
    except Exception:
        pass

    return attack


def run_attack(attack_name: str, args: argparse.Namespace) -> Path:
    try:
        from textattack import AttackArgs, Attacker
        from textattack.datasets import Dataset as TextAttackDataset
        from textattack.models.wrappers import HuggingFaceModelWrapper
    except ImportError as exc:
        raise ImportError("TextAttack is required for attack evaluation. Install requirements.txt first.") from exc

    device = get_device(args.device)
    _, _, attack_ds = load_agnews_splits(
        seed=args.seed,
        train_size=0,
        test_size=args.test_size,
        attack_size=args.attack_size,
    )
    if len(attack_ds) == 0:
        raise ValueError("--attack-size must be greater than 0 for attack evaluation.")

    model, tokenizer = load_checkpoint_model_and_tokenizer(args.checkpoint_dir)
    model.to(device)
    model.eval()

    try:
        wrapper = HuggingFaceModelWrapper(model, tokenizer, batch_size=args.attack_batch_size)
    except TypeError:
        wrapper = HuggingFaceModelWrapper(model, tokenizer)

    attack = build_attack(attack_name, wrapper, args.max_modification_rate)
    examples = [(row["text"], int(row["label"])) for row in attack_ds]
    ta_dataset = TextAttackDataset(examples)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{attack_name}-{timestamp}.txt"
    query_budget = args.query_budget or (args.max_synonyms * args.max_length)

    attack_args = AttackArgs(
        num_examples=len(examples),
        query_budget=query_budget,
        log_to_txt=str(txt_path),
        disable_stdout=not args.verbose_attack,
    )
    attacker = Attacker(attack, ta_dataset, attack_args)
    results = list(attacker.attack_dataset())
    summary = summarize_attack_results(results)

    with txt_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n===== FreeLB++ AGNews Attack Summary =====\n")
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.write(f"attack={attack_name}\n")
        handle.write(f"checkpoint_dir={args.checkpoint_dir}\n")
        handle.write(f"query_budget={query_budget}\n")
        handle.write(f"max_modification_rate={args.max_modification_rate}\n")
        handle.write(f"semantic_similarity_target={args.semantic_similarity}\n")

    return txt_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    attack_names = ["textfooler", "textbugger"] if args.attack == "both" else [args.attack]
    paths = [run_attack(attack_name, args) for attack_name in attack_names]
    print("attack_logs=" + json.dumps([str(path) for path in paths], ensure_ascii=False))


if __name__ == "__main__":
    main()
