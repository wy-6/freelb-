import argparse
import json

from common import (
    evaluate_clean,
    get_device,
    load_agnews_splits,
    load_base_model_and_tokenizer,
    set_seed,
    train_freelbpp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT with FreeLB++ on AGNews.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--checkpoint-dir", default="checkpoints/freelbpp-agnews-bert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=7000)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--adv-steps", type=int, default=30)
    parser.add_argument("--adv-lr", type=float, default=0.1)
    parser.add_argument(
        "--adv-init-mag",
        type=float,
        default=0.0,
        help="Initial perturbation magnitude. FreeLB++ can use 0 and relies on multi-step ascent.",
    )
    parser.add_argument("--skip-clean-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    train_ds, test_ds, _ = load_agnews_splits(
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        attack_size=0,
    )
    model, tokenizer = load_base_model_and_tokenizer(args.model_name)

    train_freelbpp(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        adv_steps=args.adv_steps,
        adv_lr=args.adv_lr,
        adv_init_mag=args.adv_init_mag,
    )

    if not args.skip_clean_eval:
        metrics = evaluate_clean(
            model=model,
            tokenizer=tokenizer,
            test_ds=test_ds,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
