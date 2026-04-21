from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wsrvos import WSRVOSModel, build_dataloader, evaluate, load_config, run_inference, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("WSRVOS")
    parser.add_argument("command", choices=["train", "eval", "infer"])
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load for eval/infer or resume.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_overrides = {}
    if args.output is not None:
        cli_overrides["output_dir"] = args.output
    config = load_config(args.config, cli_overrides)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = WSRVOSModel(config).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)

    if args.command == "train":
        train_loader, _ = build_dataloader(config, "train")
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        output_dir = args.output or config.output_dir
        train(config, model, train_loader, optimizer, device, output_dir)
        return

    split = getattr(config.eval, "split", "test")
    data_loader, _ = build_dataloader(config, split)
    if args.command == "eval":
        metrics = evaluate(model, data_loader, device)
        print(metrics)
    else:
        output_dir = args.output or config.output_dir
        run_inference(model, data_loader, device, output_dir)


if __name__ == "__main__":
    main()
