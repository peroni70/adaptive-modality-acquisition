"""Command line interface.

    ama run configs/patch_mnist.yaml
    ama train-value configs/patch_mnist.yaml --value-fn bit_flip
    ama eval-policy configs/patch_mnist.yaml --costs costs.npy --lambda 0.5
    ama show configs/patch_mnist.yaml --field run_dir
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import (
    run_all,
    stage_eval_policy,
    stage_greedy_order,
    stage_train_classifier,
    stage_train_value,
)
from .registry import available_apps

PER_VALUE_FN_STAGES = {
    "greedy-order": stage_greedy_order,
    "train-value": stage_train_value,
    "eval-policy": stage_eval_policy,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ama",
        description="Adaptive modality acquisition: train, evaluate, sweep.",
    )
    parser.add_argument(
        "stage",
        choices=["run", "train-classifier", "show", *PER_VALUE_FN_STAGES],
        help="pipeline stage to run ('run' does all of them; "
        "'show' prints the resolved config)",
    )
    parser.add_argument("config", help="path to a YAML experiment config")
    parser.add_argument(
        "--value-fn",
        action="append",
        dest="value_fns",
        help="value function to run; repeatable, defaults to policy.value_fns",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a config field, e.g. -o value_model.epochs=5",
    )
    parser.add_argument(
        "--costs",
        help="path to real acquisition costs (.pt/.npy/.csv): a (n_modes,) "
        "vector, or an (n_examples, n_modes) matrix of per-example costs. "
        "Replaces the simulated cost sweep; requires --lambda.",
    )
    parser.add_argument(
        "--lambda",
        dest="cost_lambda",
        type=float,
        help="exchange rate for --costs: how much you will spend per unit of "
        "expected accuracy gained, in the same units as the costs.",
    )
    parser.add_argument(
        "--device",
        help="override the device, e.g. cpu or cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--field",
        help="with 'show', print just this config field (e.g. run_dir)",
    )
    parser.add_argument(
        "--list-apps", action="store_true", help="print registered applications and exit"
    )
    return parser


def show(cfg, field: str | None) -> int:
    """Print the resolved config, so overrides can be verified before a long run."""
    import yaml

    data = cfg.to_dict()
    if field is None:
        print(yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip())
        return 0
    node = data
    for key in field.split("."):
        if not isinstance(node, dict) or key not in node:
            raise SystemExit(f"no such config field: {field}")
        node = node[key]
    print(node)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if "--list-apps" in argv:
        from . import apps  # noqa: F401  (registers the built-ins)

        print("\n".join(available_apps()))
        return 0

    args = parser.parse_args(argv)
    overrides = list(args.overrides or [])
    # Convenience flags are ordinary config overrides underneath, so they
    # compose with -o and show up in `ama show`.
    if args.costs is not None:
        overrides.append(f"policy.costs_path={args.costs}")
    if args.cost_lambda is not None:
        overrides.append(f"policy.cost_lambda={args.cost_lambda}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
    cfg = load_config(args.config, overrides)

    if cfg.policy.costs_path and cfg.policy.cost_lambda is None:
        parser.error("--costs requires --lambda (or policy.cost_lambda in the config)")

    value_fns = args.value_fns or cfg.policy.value_fns

    if args.stage == "show":
        return show(cfg, args.field)
    if args.stage == "run":
        run_all(cfg, value_fns)
    elif args.stage == "train-classifier":
        stage_train_classifier(cfg)
    else:
        stage = PER_VALUE_FN_STAGES[args.stage]
        for name in value_fns:
            stage(cfg, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
