"""``fraud-detect`` command line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .data import simulate_transactions

    df = simulate_transactions(n_cards=args.n_cards, days=args.days, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} transactions ({df['is_fraud'].mean():.3%} fraud) to {out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import train_pipeline

    cfg = load_config(args.config)
    result = train_pipeline(cfg)
    m = result.metrics
    print(f"Bundle: {result.bundle_dir}")
    print(
        f"Test PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
        f"fraud recall {m['fraud_recall']:.3f} | cost savings {m['cost_savings_pct']:.1%}"
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from .features import build_offline_features
    from .models import load_bundle
    from .pipeline import load_data, make_splits
    from .replay import replay
    from .rules import RuleEngine

    cfg = load_config(args.config)
    bundle = load_bundle(args.bundle or cfg.artifacts_dir)
    df = load_data(cfg)
    sp = make_splits(df, cfg)
    history = df.loc[sp.train.append(sp.calib).append(sp.tune)]
    stream = df.loc[sp.test]
    if args.limit:
        stream = stream.iloc[: args.limit]
    rules = RuleEngine.from_yaml(cfg.rules_path) if cfg.rules_path else None
    offline = build_offline_features(df) if args.check_parity else None
    result = replay(bundle, history, stream, rules=rules, offline_features=offline, costs=cfg.costs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(out, index=False)
    print(json.dumps(result.summary, indent=2, default=float))
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from .data import load_transactions
    from .features import build_offline_features
    from .models import load_bundle
    from .monitoring import drift_report

    cfg = load_config(args.config)
    bundle = load_bundle(args.bundle or cfg.artifacts_dir)
    df = load_transactions([args.data])
    feats = build_offline_features(df)
    X = bundle.model.model_frame(feats)
    scores = bundle.model.predict_proba(feats)
    report = drift_report(
        bundle.reference_profile, X, scores, cfg.monitoring.psi_warn, cfg.monitoring.psi_alert
    )
    print(report.to_string(index=False))
    return 1 if (report["status"] == "alert").any() and args.fail_on_alert else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app
    from .models import load_bundle
    from .rules import RuleEngine

    cfg = load_config(args.config)
    bundle = load_bundle(args.bundle or cfg.artifacts_dir)
    rules = RuleEngine.from_yaml(cfg.rules_path) if cfg.rules_path else None
    app = create_app(bundle, rules, warmup_path=args.warmup)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-detect")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="write a synthetic dataset in the Sparkov schema")
    p.add_argument("--n-cards", type=int, default=1000)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/raw/synthetic_transactions.csv")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("train", help="train, calibrate, tune thresholds and save a bundle")
    p.add_argument("--config", default="configs/default.yaml")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("replay", help="stream the test period through the online path")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--bundle")
    p.add_argument("--limit", type=int, help="only replay the first N test transactions")
    p.add_argument("--check-parity", action="store_true", help="compare online vs offline features")
    p.add_argument("--out", default="reports/replay_decisions.csv")
    p.set_defaults(func=_cmd_replay)

    p = sub.add_parser("monitor", help="PSI drift report for a new batch of transactions")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--bundle")
    p.add_argument("--data", required=True)
    p.add_argument("--fail-on-alert", action="store_true")
    p.set_defaults(func=_cmd_monitor)

    p = sub.add_parser("serve", help="start the scoring API")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--bundle")
    p.add_argument("--warmup", help="CSV of recent transactions to pre-load card history")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
