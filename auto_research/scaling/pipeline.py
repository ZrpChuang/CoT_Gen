#!/usr/bin/env python3
"""Deploy and operate the self-contained 100K continuation queue."""

import argparse
import json
import os
from pathlib import Path

from core.common import DataError, Waiting, write


def worker(args):
    from core import data, efficiency, evaluate, train, variants

    actions = {
        "audit": lambda: data.audit(),
        "inventory": lambda: train.validate_history(),
        "generate": lambda: data.generate(),
        "prepare": lambda: data.prepare(args.milestone),
        "select": lambda: variants.score(args.milestone, args.shard),
        "assemble": lambda: variants.assemble(args.milestone),
        "encode": lambda: train.encode(args.run_id, args.milestone),
        "preflight": lambda: train.preflight(args.model),
        "capacity": lambda: train.preflight(args.model, capacity=True),
        "tune": lambda: efficiency.tune(args.model, args.milestone),
        "profile": lambda: efficiency.profile(
            args.model, args.milestone, args.micro_batch, args.tag
        ),
        "train": lambda: train.train(args.run_id, args.milestone),
        "wandb": lambda: train.sync_wandb(args.run_id, args.milestone),
        "eval-preflight": lambda: evaluate.preflight(),
        "infer": lambda: evaluate.generate(args.run_id, args.milestone),
        "score": lambda: evaluate.score(args.run_id, args.milestone, args.bench),
        "verify": lambda: evaluate.verify(args.run_id, args.milestone),
    }
    result = {"exit_code": 0, "kind": args.kind}
    try:
        actions[args.kind]()
    except Waiting as exc:
        result.update(exit_code=75, error_type=type(exc).__name__, detail=str(exc))
    except (DataError, AssertionError, ValueError) as exc:
        result.update(
            exit_code=78, error_type=type(exc).__name__, detail=str(exc)[:300]
        )
        import traceback

        traceback.print_exc()
    except BaseException as exc:
        result.update(exit_code=1, error_type=type(exc).__name__)
        raise
    finally:
        if args.receipt and int(os.environ.get("RANK", "0")) == 0:
            write(args.receipt, result)
    return result["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "upgrade", "run", "supervise", "status", "stop"):
        commands.add_parser(name)
    scope = commands.add_parser("models")
    scope.add_argument("--enable", nargs="+", required=True)
    executor = commands.add_parser("execute")
    executor.add_argument("--intent", required=True)
    phase = commands.add_parser("worker")
    phase.add_argument(
        "kind",
        choices=(
            "audit",
            "inventory",
            "generate",
            "prepare",
            "select",
            "assemble",
            "encode",
            "preflight",
            "capacity",
            "tune",
            "profile",
            "train",
            "wandb",
            "eval-preflight",
            "infer",
            "score",
            "verify",
        ),
    )
    phase.add_argument("--run-id")
    phase.add_argument("--milestone", type=int)
    phase.add_argument("--model")
    phase.add_argument("--micro-batch", type=int, default=1)
    phase.add_argument("--tag", default="manual")
    phase.add_argument("--shard", type=int)
    phase.add_argument("--bench")
    phase.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.command == "execute":
        from core.queue import execute

        return execute(args.intent)
    if args.command == "worker":
        return worker(args)
    from core import queue

    if args.command == "models":
        queue.set_models(args.enable)
        return 0

    value = {
        "start": queue.start,
        "upgrade": queue.upgrade,
        "run": queue.run,
        "supervise": queue.supervise,
        "status": queue.status,
        "stop": queue.halt,
    }[args.command]()
    if value is not None:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
