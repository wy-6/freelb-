import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


AGNEWS_LABELS = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_agnews_splits(
    seed: int,
    train_size: int = 20000,
    test_size: int = 7000,
    attack_size: int = 1000,
) -> Tuple[Dataset, Dataset, Dataset]:
    raw = load_dataset("ag_news")
    train_pool = raw["train"].shuffle(seed=seed)
    test_pool = raw["test"].shuffle(seed=seed)

    if train_size > len(train_pool):
        raise ValueError(f"--train-size {train_size} exceeds AGNews train size {len(train_pool)}.")
    if test_size > len(test_pool):
        raise ValueError(f"--test-size {test_size} exceeds AGNews test size {len(test_pool)}.")
    if attack_size > test_size:
        raise ValueError("--attack-size must be less than or equal to --test-size.")

    train_ds = train_pool.select(range(train_size))
    test_ds = test_pool.select(range(test_size))
    attack_ds = test_ds.select(range(attack_size))
    return train_ds, test_ds, attack_ds


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize_batch(batch: Dict[str, List]) -> Dict[str, List]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text", "label"])
    tokenized.set_format(type="torch")
    return tokenized


def build_dataloader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def load_base_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=AGNEWS_LABELS)
    return model, tokenizer


def load_checkpoint_model_and_tokenizer(
    checkpoint_dir: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=AGNEWS_LABELS)
    return model, tokenizer


def mask_delta(delta: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return delta * attention_mask.unsqueeze(-1).to(delta.dtype)


def init_delta(embeds: torch.Tensor, attention_mask: torch.Tensor, init_mag: float) -> torch.Tensor:
    delta = torch.zeros_like(embeds)
    if init_mag <= 0:
        return delta

    delta.uniform_(-1.0, 1.0)
    delta = mask_delta(delta, attention_mask)
    dims = attention_mask.sum(dim=1).clamp(min=1).to(delta.dtype) * embeds.size(-1)
    mag = init_mag / torch.sqrt(dims)
    return delta * mag.view(-1, 1, 1)


def freelbpp_train_step(
    model: AutoModelForSequenceClassification,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    adv_steps: int,
    adv_lr: float,
    adv_init_mag: float,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = batch.pop("input_ids")
    attention_mask = batch.get("attention_mask")
    embeds = model.get_input_embeddings()(input_ids)

    delta = init_delta(embeds, attention_mask, adv_init_mag).detach()
    delta.requires_grad_()

    total_loss = 0.0
    steps = max(adv_steps, 1)
    for step in range(steps):
        outputs = model(inputs_embeds=embeds + delta, **batch)
        loss = outputs.loss / steps
        total_loss += loss.item()
        loss.backward()

        if step == steps - 1:
            break

        grad = delta.grad.detach()
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).clamp(min=1e-8)
        delta = (delta + adv_lr * grad / grad_norm.view(-1, 1, 1)).detach()
        delta = mask_delta(delta, attention_mask)
        delta.requires_grad_()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    return total_loss


def train_freelbpp(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    checkpoint_dir: str,
    device: torch.device,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    adv_steps: int,
    adv_lr: float,
    adv_init_mag: float,
) -> None:
    tokenized_train = tokenize_dataset(train_ds, tokenizer, max_length)
    train_loader = build_dataloader(tokenized_train, tokenizer, batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.to(device)
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = freelbpp_train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                adv_steps=adv_steps,
                adv_lr=adv_lr,
                adv_init_mag=adv_init_mag,
                max_grad_norm=max_grad_norm,
            )
            running_loss += loss
            progress.set_postfix(loss=f"{running_loss / max(progress.n, 1):.4f}")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


@torch.no_grad()
def evaluate_clean(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    test_ds: Dataset,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Dict[str, float]:
    tokenized_test = tokenize_dataset(test_ds, tokenizer, max_length)
    test_loader = build_dataloader(tokenized_test, tokenizer, batch_size, shuffle=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    for batch in tqdm(test_loader, desc="clean eval"):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"clean_accuracy": correct / total if total else 0.0, "total": total}


def result_kind(result) -> str:
    return result.__class__.__name__.lower()


def extract_query_count(result) -> int:
    candidates = [
        getattr(result, "num_queries", None),
        getattr(getattr(result, "perturbed_result", None), "num_queries", None),
        getattr(getattr(result, "original_result", None), "num_queries", None),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def summarize_attack_results(results: Iterable) -> Dict[str, float]:
    total = 0
    successful = 0
    failed = 0
    skipped = 0
    queries = []

    for result in results:
        total += 1
        kind = result_kind(result)
        if "successful" in kind:
            successful += 1
        elif "failed" in kind:
            failed += 1
        elif "skipped" in kind:
            skipped += 1
        queries.append(extract_query_count(result))

    attempted = successful + failed
    return {
        "total": total,
        "attempted": attempted,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "accuracy_under_attack": failed / total if total else 0.0,
        "attack_success_rate": successful / attempted if attempted else 0.0,
        "avg_queries": float(np.mean(queries)) if queries else 0.0,
    }
