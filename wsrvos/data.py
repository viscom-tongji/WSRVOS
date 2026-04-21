from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.a2d_sentences.a2d_sentences_dataset import A2DSentencesDataset
from datasets.jhmdb_sentences.jhmdb_sentences_dataset import JHMDBSentencesDataset
from datasets.refer_youtube_vos.refer_youtube_vos_dataset import ReferYouTubeVOSDataset
from misc import nested_tensor_from_videos_list

from .augmentations import TextAugmentationBank


@dataclass
class SampleMeta:
    sample_id: str
    dataset_name: str
    text: str
    has_masks: bool


def _extract_target_info(targets: List[Any]) -> Dict[str, Any]:
    frame_masks: List[torch.Tensor | None] = []
    frame_valid: List[bool] = []
    for target in targets:
        if target is None or "masks" not in target:
            frame_masks.append(None)
            frame_valid.append(False)
            continue
        frame_masks.append(target["masks"][0].to(torch.float32))
        frame_valid.append(True)
    return {
        "frame_masks": frame_masks,
        "frame_valid": torch.tensor(frame_valid, dtype=torch.bool),
        "targets": targets,
    }


class WeakRVOSDataset(Dataset):
    def __init__(self, config, split: str, augmentation_bank: TextAugmentationBank | None = None) -> None:
        super().__init__()
        dataset_name = config.dataset.name
        common_kwargs = dict(
            horizontal_flip_augmentations=config.data.horizontal_flip,
            resize_and_crop_augmentations=config.data.resize_and_crop,
            random_color=config.data.random_color,
            train_short_size=config.data.train_short_size,
            train_max_size=config.data.train_max_size,
            eval_short_size=config.data.eval_short_size,
            eval_max_size=config.data.eval_max_size,
            window_size=config.data.window_size,
            frame_interval=getattr(config.data, "frame_interval", 10),
            dataset_coco_gt_format_path=getattr(config.dataset, "gt_annotations", None),
            distributed=False,
        )
        if dataset_name == "a2d_sentences":
            self.base = A2DSentencesDataset(split, dataset_path=config.dataset.root, **common_kwargs)
        elif dataset_name == "jhmdb_sentences":
            self.base = JHMDBSentencesDataset(split, dataset_path=config.dataset.root, **common_kwargs)
        elif dataset_name == "ref_youtube_vos":
            self.base = ReferYouTubeVOSDataset(split, dataset_path=config.dataset.root, **common_kwargs)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.split = split
        self.augmentation_bank = augmentation_bank

    def build_sample_id(self, idx: int) -> str:
        if self.dataset_name == "a2d_sentences":
            _, video_id, frame_idx, instance_id = self.base.text_annotations[idx]
            return f"a2d:{video_id}:{frame_idx}:{instance_id}"
        if self.dataset_name == "jhmdb_sentences":
            video_id, frame_path, _, _, _ = self.base.samples_metadata[idx]
            frame_idx = frame_path.split("/")[-1].split(".")[0]
            return f"jhmdb:{video_id}:{frame_idx}"
        video_id, frame_indices, exp_dict = self.base.samples_list[idx]
        exp_id = exp_dict.get("exp_id", idx)
        return f"ytvos:{video_id}:{exp_id}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.build_sample_id(idx)
        item = self.base[idx]
        if self.split == "train":
            frames, targets, text = item
            metadata = None
        else:
            if self.dataset_name == "ref_youtube_vos":
                frames, metadata, targets, text = item
            else:
                frames, targets, text = item
                metadata = None
        text = " ".join(text.lower().split())
        if self.augmentation_bank is not None and self.split == "train":
            aug = self.augmentation_bank.get(sample_id, text)
        else:
            aug = {
                "original_text": text,
                "positive_texts": [text],
                "positive_confidences": [1.0],
                "negative_texts": [],
            }
        target_info = _extract_target_info(list(targets))
        return {
            "frames": frames,
            "metadata": metadata,
            "sample_id": sample_id,
            "dataset_name": self.dataset_name,
            "original_text": aug["original_text"],
            "positive_texts": aug["positive_texts"],
            "positive_confidences": aug["positive_confidences"],
            "negative_texts": aug["negative_texts"],
            "frame_masks": target_info["frame_masks"],
            "frame_valid": target_info["frame_valid"],
            "targets": target_info["targets"],
        }

    def __len__(self) -> int:
        return len(self.base)


def collate_weak_rvos(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = [item["frames"] for item in batch]
    nested = nested_tensor_from_videos_list(videos)
    return {
        "videos": nested,
        "sample_ids": [item["sample_id"] for item in batch],
        "dataset_names": [item["dataset_name"] for item in batch],
        "original_texts": [item["original_text"] for item in batch],
        "positive_texts": [item["positive_texts"] for item in batch],
        "positive_confidences": [item["positive_confidences"] for item in batch],
        "negative_texts": [item["negative_texts"] for item in batch],
        "frame_masks": [item["frame_masks"] for item in batch],
        "frame_valid": torch.stack([item["frame_valid"] for item in batch], dim=0),
        "targets": [item["targets"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def build_dataloader(config, split: str) -> Tuple[DataLoader, WeakRVOSDataset]:
    augmentation_bank = None
    if split == "train":
        augmentation_bank = TextAugmentationBank(
            num_positive=config.model.num_positive,
            num_negative=config.model.num_negative,
            json_path=getattr(config.dataset, "augmentation_file", None),
            seed=config.train.seed,
        )
    dataset = WeakRVOSDataset(config, split, augmentation_bank)
    if augmentation_bank is not None:
        augmentation_bank.register_corpus(getattr(dataset.base, "text_annotations", []) and [a[0] for a in getattr(dataset.base, "text_annotations", [])] or [])
        if not augmentation_bank.corpus:
            corpus = []
            if hasattr(dataset.base, "samples_list"):
                for _, _, exp_dict in dataset.base.samples_list:
                    corpus.append(exp_dict["exp"])
            elif hasattr(dataset.base, "samples_metadata"):
                for _, _, _, _, text in dataset.base.samples_metadata:
                    corpus.append(text)
            augmentation_bank.register_corpus(corpus)
    max_train_samples = getattr(config.data, "max_train_samples", None)
    if split == "train" and max_train_samples is not None:
        dataset = Subset(dataset, range(min(int(max_train_samples), len(dataset))))

    shuffle = split == "train"
    batch_size = config.train.batch_size if split == "train" else config.eval.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_weak_rvos,
    )
    return loader, dataset
