"""Pipeline stages.

The experiment runs as four stages, each reading the previous one's artifacts
from the run directory:

    train-classifier -> greedy-order -> train-value -> eval-policy

Stages are separate so a sweep over value functions can share one classifier,
and so a long policy evaluation can be re-run without retraining anything.
Models are stored as state dicts and rebuilt from the config, so a checkpoint
never depends on where a class happens to live in the source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build import (
    build_classifier,
    build_masker_and_metrics,
    build_value_model,
    example_shape,
    masked_loaders,
    value_loaders,
)
from .config import Config
from .costs import CostModel, load_costs
from .data import PolicyDataset
from .diagnostics import model_confusion_rates
from .evaluation import (
    evaluate_by_prefix,
    evaluate_policy,
    evaluate_static_baselines,
    greedy_modality_order,
    greedy_prefix_gains,
)
from .modeling import TemperatureScaled, fit_temperature
from .registry import load_splits
from .results import ResultWriter
from .training import (
    constant_predictor_mse,
    eval_classifier,
    mean_target,
    regression_skill,
    train_value_model,
    train_classifier,
)
from .value_fns import get_value_fn


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def value_dir(cfg: Config, value_fn: str) -> Path:
    return cfg.run_path / value_fn


def classifier_path(cfg: Config) -> Path:
    return cfg.run_path / "classifier.pt"


def _load_classifier(cfg, masker, metrics, splits) -> torch.nn.Module:
    path = classifier_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"no classifier at {path}; run the train-classifier stage first"
        )
    model = build_classifier(
        cfg.classifier, masker, metrics, example_shape(splits["train"])
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(cfg.resolve_device()).eval()


def stage_train_classifier(cfg: Config) -> dict:
    """Train the masked classifier that every later stage treats as fixed."""
    set_seed(cfg.seed)
    masker, metrics = build_masker_and_metrics(cfg)
    splits = load_splits(cfg)
    loaders = masked_loaders(splits, masker, cfg)
    model = build_classifier(
        cfg.classifier, masker, metrics, example_shape(splits["train"])
    )

    print(f"Training classifier on {len(splits['train'])} examples...")
    model, val_acc = train_classifier(
        model,
        loaders["train"],
        loaders["val"],
        n_epochs=cfg.classifier.epochs,
        lr=cfg.classifier.lr,
        weight_decay=cfg.classifier.weight_decay,
        device=cfg.resolve_device(),
    )
    cfg.run_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), classifier_path(cfg))

    test = eval_classifier(model, loaders["test"], metrics, device=cfg.resolve_device())
    print(f"Classifier: val acc {val_acc:.4f} | test {test}")
    (cfg.run_path / "classifier_metrics.json").write_text(
        json.dumps({"val_accuracy": val_acc, **test}, indent=2)
    )
    return test


def stage_confusion_rate(cfg: Config, split: str = "train") -> dict:
    """Measure how often acquiring confuses the classifier.

    Reported before any value model is trained, because the answer is what
    decides which value function can represent the problem.
    """
    set_seed(cfg.seed)
    masker, metrics = build_masker_and_metrics(cfg)
    splits = load_splits(cfg)
    if split not in splits:
        raise ValueError(f"no split named {split!r}; have {sorted(splits)}")
    classifier = _load_classifier(cfg, masker, metrics, splits)

    print(f"Sampling acquisitions on the {split} split...")
    rates = model_confusion_rates(
        classifier,
        splits[split],
        masker,
        cfg.context_idx,
        n_repeats=cfg.value_model.n_repeats.get(split, 1),
        batch_size=cfg.policy.batch_size,
        seed=cfg.seed,
        device=cfg.resolve_device(),
    )
    print()
    print(rates.report())

    cfg.run_path.mkdir(parents=True, exist_ok=True)
    out = cfg.run_path / f"confusion_rates_{split}.json"
    out.write_text(json.dumps({"split": split, **rates.to_dict()}, indent=2))
    print(f"\nWrote {out}")
    return rates.to_dict()


def stage_greedy_order(cfg: Config, value_fn_name: str) -> list[int]:
    """Rank modalities by greedy validation gain, the static baseline order."""
    set_seed(cfg.seed)
    masker, metrics = build_masker_and_metrics(cfg)
    splits = load_splits(cfg)
    model = _load_classifier(cfg, masker, metrics, splits)
    value_fn = get_value_fn(value_fn_name)

    print(f"Computing greedy modality order ({value_fn_name})...")
    order = greedy_modality_order(
        model, splits["val"], masker, cfg.context_idx, value_fn,
        device=cfg.resolve_device(),
    )
    baseline, gains = greedy_prefix_gains(
        model, splits["val"], masker, order, cfg.context_idx,
        device=cfg.resolve_device(),
    )
    prefix_rows = evaluate_by_prefix(
        model, splits["test"], masker, order, metrics, cfg.context_idx,
        device=cfg.resolve_device(),
    )

    out = value_dir(cfg, value_fn_name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "greedy_order.json").write_text(
        json.dumps(
            {
                "order": order,
                "baseline_val_accuracy": baseline,
                "prefix_val_gains": gains,
                "test_by_prefix": prefix_rows,
            },
            indent=2,
        )
    )
    print(f"Greedy order: {order}")
    return order


def _read_greedy(cfg: Config, value_fn_name: str) -> dict:
    path = value_dir(cfg, value_fn_name) / "greedy_order.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no greedy order at {path}; run the greedy-order stage first"
        )
    return json.loads(path.read_text())


def stage_train_value(cfg: Config, value_fn_name: str):
    """Train, and optionally calibrate, the value model for one value function."""
    set_seed(cfg.seed)
    masker, metrics = build_masker_and_metrics(cfg)
    splits = load_splits(cfg)
    classifier = _load_classifier(cfg, masker, metrics, splits)
    value_fn = get_value_fn(value_fn_name)
    vcfg = cfg.value_model

    loaders = value_loaders(splits, masker, classifier, value_fn, cfg)
    model = build_value_model(
        vcfg, masker, value_fn, example_shape(splits["train"])
    )
    out = value_dir(cfg, value_fn_name)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training value model ({value_fn_name})...")
    model, best_score, val_scores = train_value_model(
        model,
        loaders["train"],
        loaders["val"],
        value_fn,
        masker,
        cfg.context_idx,
        n_epochs=vcfg.epochs,
        lr=vcfg.lr,
        weight_decay=vcfg.weight_decay,
        use_hinge_loss=vcfg.use_hinge_loss,
        hinge_lam=vcfg.hinge_lam,
        checkpoint_path=out / "value_model.pt",
        device=cfg.resolve_device(),
    )
    report = {"best_score": best_score, "val_scores": val_scores}
    if not value_fn.higher_is_better:
        # A regression-valued target is only useful if it beats predicting the
        # training mean, so record that comparison rather than the MSE alone.
        constant = mean_target(loaders["train"])
        baseline = constant_predictor_mse(loaders["val"], constant)
        report.update(
            {
                "train_mean_target": constant,
                "constant_predictor_val_mse": baseline,
                "val_mse": best_score,
                "skill_vs_constant": regression_skill(best_score, baseline),
            }
        )
        skill = report["skill_vs_constant"]
        print(
            f"Val MSE {best_score:.6f} vs constant-predictor {baseline:.6f}: "
            f"{skill:+.1%} relative error reduction (skill {skill:+.4f})"
        )
    (out / "value_model_scores.json").write_text(json.dumps(report, indent=2))

    if vcfg.calibrate and value_fn.is_probabilistic:
        print("Calibrating...")
        calibrated = fit_temperature(
            model, loaders["val"], value_fn,
            n_epochs=vcfg.calibration_epochs, device=cfg.resolve_device(),
        )
        torch.save(
            {"temperature": calibrated.temperature.detach().cpu()},
            out / "calibration.pt",
        )
        print(f"Temperature: {calibrated.temperature.item():.4f}")
    print(f"Best validation {value_fn.name} score: {best_score:.4f}")
    return model


def _load_value_models(cfg, masker, value_fn, splits) -> list[tuple[str, torch.nn.Module]]:
    """Load the trained value model, plus its calibrated twin when present."""
    out = value_dir(cfg, value_fn.name)
    path = out / "value_model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"no value model at {path}; run the train-value stage first"
        )
    model = build_value_model(
        cfg.value_model, masker, value_fn, example_shape(splits["train"])
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    variants = [("uncal", model)]

    calib = out / "calibration.pt"
    if cfg.policy.include_calibrated and calib.exists():
        temperature = torch.load(calib, map_location="cpu")["temperature"]
        wrapped = TemperatureScaled(model)
        with torch.no_grad():
            wrapped.temperature.copy_(temperature)
        variants.append(("cal", wrapped.eval()))
    return variants


def _cost_settings(cfg: Config, n_modes: int, device: str):
    """The cost settings to evaluate, as ``(label, CostModel)`` pairs.

    Real costs give one setting; simulated costs give one per alpha, tracing
    out how each policy responds as acquisition grows more expensive.
    """
    policy = cfg.policy
    if policy.costs_path:
        if policy.cost_lambda is None:
            raise ValueError(
                "policy.cost_lambda is required when policy.costs_path is set: "
                "it converts costs into value units"
            )
        costs = load_costs(policy.costs_path)
        model = CostModel(
            n_modes, costs=costs, lambda_=policy.cost_lambda, device=device
        )
        print(f"Using costs from {policy.costs_path} ({model}), lambda={policy.cost_lambda}")
        return [(f"lambda_{policy.cost_lambda}", model)]

    return [
        (
            f"alpha_{alpha}",
            CostModel(
                n_modes,
                mean_costs=np.ones(n_modes),
                cov_costs=np.eye(n_modes),
                scale=alpha,
                device=device,
            ),
        )
        for alpha in policy.alphas
    ]


def _policy_runs(cfg: Config, value_fn) -> list[dict]:
    """Enumerate the (value model, optimizer) runs the config asks for."""
    runs = []
    for optimizer in cfg.policy.optimizers:
        runs.append({"optimizer": optimizer, "reverse": False})
    if cfg.policy.include_reverse and value_fn.name == "acc_change":
        runs.append({"optimizer": "hybrid_usm", "reverse": True})
    return runs


def stage_eval_policy(cfg: Config, value_fn_name: str) -> None:
    """Sweep costs, and score every policy against the static greedy baselines."""
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    masker, metrics = build_masker_and_metrics(cfg)
    splits = load_splits(cfg)
    classifier = _load_classifier(cfg, masker, metrics, splits)
    value_fn = get_value_fn(value_fn_name)
    greedy = _read_greedy(cfg, value_fn_name)
    variants = _load_value_models(cfg, masker, value_fn, splits)

    n_modes = masker.n_modes
    out = value_dir(cfg, value_fn_name)
    test = splits["test"]

    for label, cost_model in _cost_settings(cfg, n_modes, device):
        print(f"\n=== {value_fn_name} | {label} ===")
        with ResultWriter(
            out / f"results_{label}.csv",
            out / f"samples_{label}.csv",
            value_fn_name,
            label,
        ) as writer:
            baselines, base_acc, base_rows, base_correct = evaluate_static_baselines(
                classifier, test, masker, cfg.context_idx, greedy["order"], metrics,
                cost_model=cost_model,
                batch_size=cfg.policy.batch_size, device=device,
            )
            for name, summary in baselines:
                writer.write_summary(name, summary)
            writer.write_samples(base_rows)

            for variant_name, value_model in variants:
                for run in _policy_runs(cfg, value_fn):
                    name = f"{variant_name}_{run['optimizer']}"
                    if run["reverse"]:
                        name += "_reverse"
                    dataset = PolicyDataset(
                        test, masker, cfg.context_idx,
                        value_models=value_model, value_fn=value_fn,
                        method="eama", opt_method=run["optimizer"],
                        cost_model=cost_model,
                        deterministic=True, reverse=run["reverse"],
                        single_stage=cfg.policy.single_stage,
                        device=device, return_num_stages=True,
                    )
                    loader = DataLoader(
                        dataset, batch_size=cfg.policy.batch_size, shuffle=False
                    )
                    summary, rows = evaluate_policy(
                        classifier, loader, cfg.context_idx, n_modes, metrics,
                        baseline_accuracy=base_acc, baseline_correct=base_correct,
                        method_name=name, device=device,
                    )
                    print(
                        f"{name}: acc {summary['accuracy']:.4f} | "
                        f"cost {summary['cost']:.4f} | reward {summary['reward']:.4f}"
                    )
                    writer.write_summary(name, summary)
                    writer.write_samples(rows)

            if cfg.policy.include_adaptive_greedy:
                dataset = PolicyDataset(
                    test, masker, cfg.context_idx, method="adaptive_greedy",
                    cost_model=cost_model,
                    deterministic=True, greedy_order=greedy["order"],
                    greedy_prefix_gains=greedy["prefix_val_gains"],
                    device=device, return_num_stages=True,
                )
                loader = DataLoader(
                    dataset, batch_size=cfg.policy.batch_size, shuffle=False
                )
                summary, rows = evaluate_policy(
                    classifier, loader, cfg.context_idx, n_modes, metrics,
                    baseline_accuracy=base_acc, baseline_correct=base_correct,
                    method_name="adaptive_greedy", device=device,
                )
                print(
                    f"adaptive_greedy: acc {summary['accuracy']:.4f} | "
                    f"cost {summary['cost']:.4f} | reward {summary['reward']:.4f}"
                )
                writer.write_summary("adaptive_greedy", summary)
                writer.write_samples(rows)
        print(f"Wrote {out / f'results_{label}.csv'}")


def run_all(cfg: Config, value_fns: list[str] | None = None) -> None:
    """Run every stage, for each requested value function."""
    value_fns = value_fns or cfg.policy.value_fns
    if not classifier_path(cfg).exists():
        stage_train_classifier(cfg)
    else:
        print(f"Using existing classifier at {classifier_path(cfg)}")
    for name in value_fns:
        stage_greedy_order(cfg, name)
        stage_train_value(cfg, name)
        stage_eval_policy(cfg, name)
