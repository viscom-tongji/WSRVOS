from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _move_nested_to_device(batch, device):
    batch["videos"] = batch["videos"].to(device)
    return batch


def _resize_predictions(pred_masks: torch.Tensor, frame_masks, frame_valid) -> Dict[str, float]:
    intersections = 0.0
    unions = 0.0
    mean_ious = []
    for sample_idx in range(pred_masks.shape[0]):
        for frame_idx in range(pred_masks.shape[1]):
            if not frame_valid[sample_idx, frame_idx]:
                continue
            gt_mask = frame_masks[sample_idx][frame_idx]
            pred = pred_masks[sample_idx, frame_idx : frame_idx + 1].unsqueeze(0)
            pred = F.interpolate(pred, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            pred = pred.sigmoid() > 0.5
            gt = gt_mask > 0.5
            intersection = (pred & gt).float().sum().item()
            union = (pred | gt).float().sum().item()
            iou = intersection / max(union, 1.0)
            intersections += intersection
            unions += union
            mean_ious.append(iou)
    overall_iou = intersections / max(unions, 1.0)
    mean_iou = sum(mean_ious) / max(len(mean_ious), 1)
    return {"overall_iou": overall_iou, "mean_iou": mean_iou}


def train(config, model, train_loader, optimizer, device, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    amp_enabled = bool(config.train.amp) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        autocast_context = lambda: torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_context = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)
    best_loss = float("inf")
    max_steps_per_epoch = getattr(config.train, "max_steps_per_epoch", None)

    for epoch in range(config.train.epochs):
        model.train()
        running = {"loss": 0.0, "steps": 0}
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{config.train.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = _move_nested_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(batch)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += loss.item()
            running["steps"] += 1
            progress.set_postfix(loss=f"{running['loss'] / running['steps']:.4f}")
            if max_steps_per_epoch is not None and step >= int(max_steps_per_epoch):
                break

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, output_path / "last.pth")
        avg_loss = running["loss"] / max(running["steps"], 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, output_path / "best.pth")


@torch.no_grad()
def evaluate(model, data_loader, device) -> Dict[str, float]:
    model.eval()
    results = []
    for batch in tqdm(data_loader, desc="eval"):
        batch = _move_nested_to_device(batch, device)
        outputs = model(batch)
        results.append(_resize_predictions(outputs["pred_masks"], batch["frame_masks"], batch["frame_valid"]))
    if not results:
        return {"overall_iou": 0.0, "mean_iou": 0.0}
    return {
        "overall_iou": sum(item["overall_iou"] for item in results) / len(results),
        "mean_iou": sum(item["mean_iou"] for item in results) / len(results),
    }


@torch.no_grad()
def run_inference(model, data_loader, device, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="infer")):
        batch = _move_nested_to_device(batch, device)
        outputs = model(batch)
        pred_masks = outputs["pred_masks"].sigmoid().cpu()
        for sample_idx, sample_id in enumerate(batch["sample_ids"]):
            sample_dir = output_path / sample_id.replace(":", "_")
            sample_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx, mask in enumerate(pred_masks[sample_idx]):
                mask = (mask > 0.5).to(torch.uint8).numpy() * 255
                from PIL import Image

                Image.fromarray(mask).save(sample_dir / f"{frame_idx:05d}.png")

    with (output_path / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "done"}, handle, indent=2)
