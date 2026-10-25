"""Browser lab and reproducible command-line training/evaluation."""

import argparse
import json
from pathlib import Path
import time


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="AUV Practice: watch and train TD3 locally")
    commands = root.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Run the local browser interface (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", type=Path, default=Path("runs"))
    train = commands.add_parser("train", help="Train headlessly, evaluate, and save a checkpoint")
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--scenario", choices=["open", "reef", "channel"], default="reef")
    train.add_argument("--current", type=float, default=0.08)
    train.add_argument("--noise", type=float, default=0.15)
    train.add_argument("--eval-episodes", type=int, default=10)
    train.add_argument("--data-dir", type=Path, default=Path("runs"))
    train.add_argument("--load", help="Name of an existing checkpoint in --data-dir")
    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint or map-aware baseline")
    evaluate.add_argument("--checkpoint", help="Checkpoint filename in --data-dir")
    evaluate.add_argument("--controller", choices=["td3", "baseline", "random"], default="td3")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--scenario", choices=["open", "reef", "channel"], default="reef")
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--current", type=float, default=0.08)
    evaluate.add_argument("--data-dir", type=Path, default=Path("runs"))
    return root


def wait_with_progress(lab) -> dict:
    last_print = 0.0
    while lab.worker and lab.worker.is_alive():
        state = lab.wait(timeout=0.25)
        now = time.monotonic()
        if now - last_print > 5:
            t = state["training"]
            print(f"{state['mode']}: {t['steps']:,}/{t['total_steps']:,} steps | "
                  f"{t['episodes']} episodes | {t['updates']} updates | "
                  f"{t['steps_per_second']:.0f} steps/s", flush=True)
            last_print = now
    state = lab.state()
    if state["mode"] == "error":
        raise RuntimeError(state["message"])
    return state


def main() -> None:
    args = parser().parse_args()
    if args.command in {None, "serve"}:
        import uvicorn
        from .server import create_app
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        print(f"Open http://{host}:{port} in your browser. Ctrl+C stops the simulator.", flush=True)
        uvicorn.run(create_app(getattr(args, "data_dir", Path("runs"))), host=host, port=port)
        return

    from .experiment import Experiment
    lab = Experiment(args.data_dir, seed=args.seed, scenario=args.scenario, current_strength=args.current)
    try:
        if args.command == "train":
            if args.load:
                lab.load(args.load)
                print("Loaded checkpoint configuration; replay will refill.", flush=True)
            lab.start_training(args.steps, args.noise)
            wait_with_progress(lab)
            lab.start_evaluation(args.eval_episodes, "td3")
            state = wait_with_progress(lab)
            lab.save()
        else:
            if args.checkpoint:
                lab.load(args.checkpoint)
                print("Using the checkpoint's scenario/current/seed configuration.", flush=True)
            elif args.controller == "td3":
                print("No checkpoint: evaluating a fresh, untrained TD3 actor.", flush=True)
            lab.start_evaluation(args.episodes, args.controller)
            state = wait_with_progress(lab)
        output = args.data_dir / "latest-results.json"
        lab.export(output)
        print(json.dumps(state["evaluation"], indent=2), flush=True)
        print(f"Results: {output.resolve()}", flush=True)
        if args.command == "train":
            print(lab.state()["message"], flush=True)
    finally:
        lab.close()


if __name__ == "__main__":
    main()
