from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

from models.video_swin_transformer import SwinTransformer3D, compute_mask, configs, get_window_size
from .losses import classification_loss, masked_mean, segmentation_loss, temporal_ranking_loss


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50265, max_length: int = 40) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 3

    def _encode_text(self, text: str) -> List[int]:
        tokens = text.lower().split()
        ids = [self.bos_token_id]
        for token in tokens[: self.max_length - 2]:
            token_hash = abs(hash(token)) % max(self.vocab_size - 4, 1)
            ids.append(token_hash + 4)
        ids.append(self.eos_token_id)
        return ids

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        encoded = [self._encode_text(text) for text in texts]
        max_len = max(len(ids) for ids in encoded)
        max_len = min(max_len, self.max_length)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attn(query, memory, memory, key_padding_mask=memory_padding_mask)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return query


class WSRVOSModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_positive = config.model.num_positive
        self.num_negative = config.model.num_negative
        self.max_expressions = self.num_positive + self.num_negative
        self.selection_checkpoints = list(config.model.selection_layers)
        self.selection_topk_visual = config.model.selection_topk_visual
        self.selection_topk_text = config.model.selection_topk_text
        self.mask_threshold = config.model.mask_threshold
        self.hidden_dim = config.model.hidden_dim
        self.ranking_epsilon = config.loss.ranking_epsilon
        self.lambda_seg = config.loss.lambda_seg
        self.lambda_tmp = config.loss.lambda_tmp

        backbone_cfg = dict(configs[config.model.visual_backbone])
        backbone_cfg["use_checkpoint"] = getattr(config.model, "use_checkpoint", False)
        self.visual_encoder = SwinTransformer3D(**backbone_cfg)
        self._load_visual_pretrained(getattr(config.model, "visual_pretrained", None))
        self.visual_embed_dim = backbone_cfg["embed_dim"]
        self.visual_stage_dims = [
            self.visual_embed_dim * 2,
            self.visual_embed_dim * 4,
            self.visual_embed_dim * 4,
            self.visual_embed_dim * 8,
        ]

        self.text_encoder, self.tokenizer = self._build_text_components(config.model.text_encoder)
        text_hidden = self.text_encoder.config.hidden_size

        self.selection_text_to_visual = nn.ModuleList(
            nn.Linear(text_hidden, dim) for dim in self.visual_stage_dims
        )
        self.selection_visual_to_text = nn.ModuleList(
            nn.Linear(dim, text_hidden) for dim in self.visual_stage_dims
        )

        final_visual_dim = self.visual_stage_dims[-1]
        self.video_proj = nn.Linear(final_visual_dim, self.hidden_dim)
        self.text_proj = nn.Linear(text_hidden, self.hidden_dim)
        self.video_interaction = CrossAttentionBlock(self.hidden_dim, config.model.num_heads)
        self.text_interaction = CrossAttentionBlock(self.hidden_dim, config.model.num_heads)
        self.segmentation_flow = nn.Linear(self.hidden_dim, self.max_expressions)

        if getattr(config.model, "freeze_visual_encoder", True):
            for param in self.visual_encoder.parameters():
                param.requires_grad_(False)
        if getattr(config.model, "freeze_text_encoder", True):
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)

    def _load_visual_pretrained(self, checkpoint_path: str | None) -> None:
        if not checkpoint_path:
            return
        checkpoint_path = str(checkpoint_path)
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            import os

            if not os.path.isfile(checkpoint_path):
                return
        checkpoint_file = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_file.get("state_dict", checkpoint_file)
        state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        if "patch_embed.proj.weight" in state_dict and state_dict["patch_embed.proj.weight"].dim() == 5:
            state_dict["patch_embed.proj.weight"] = state_dict["patch_embed.proj.weight"].sum(dim=2, keepdim=True)
        self.visual_encoder.load_state_dict(state_dict, strict=False)

    def _build_text_components(self, model_path: str) -> Tuple[RobertaModel, RobertaTokenizerFast]:
        try:
            model = RobertaModel.from_pretrained(model_path)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
            return model, tokenizer
        except Exception:
            try:
                model = RobertaModel.from_pretrained("roberta-base")
                tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
                return model, tokenizer
            except Exception:
                config = RobertaConfig()
                model = RobertaModel(config)
                tokenizer = SimpleTokenizer(vocab_size=config.vocab_size, max_length=40)
                return model, tokenizer

    def _prepare_expressions(
        self,
        original_texts: List[str],
        positive_texts: List[List[str]],
        negative_texts: List[List[str]],
        positive_confidences: List[List[float]],
        device: torch.device,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        expression_texts = []
        labels = []
        valid_mask = []
        confidences = []
        for original, positives, negatives, scores in zip(
            original_texts,
            positive_texts,
            negative_texts,
            positive_confidences,
        ):
            pos = list(positives[: self.num_positive]) or [original]
            pos_scores = list(scores[: self.num_positive]) or [1.0]
            while len(pos) < self.num_positive:
                pos.append(original)
                pos_scores.append(1.0)
            neg = list(negatives[: self.num_negative])
            while len(neg) < self.num_negative:
                neg.append("")
            expression_texts.extend(pos + neg)
            labels.append([1.0] * self.num_positive + [0.0] * self.num_negative)
            valid_mask.append([1] * self.num_positive + [int(text != "") for text in neg])
            confidences.append(pos_scores)
        return (
            expression_texts,
            torch.tensor(labels, dtype=torch.float32, device=device),
            torch.tensor(valid_mask, dtype=torch.bool, device=device),
            torch.tensor(confidences, dtype=torch.float32, device=device),
        )

    def _tokenize(
        self,
        texts: List[str],
        batch_size: int,
        num_expressions: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = self.text_encoder.embeddings(input_ids=encoded["input_ids"])
        attention_mask = encoded["attention_mask"].view(batch_size, num_expressions, -1)
        return hidden.view(batch_size, num_expressions, hidden.shape[1], hidden.shape[2]), attention_mask

    def _advance_text(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        batch_size, num_expressions, seq_len, dim = hidden.shape
        hidden = hidden.view(batch_size * num_expressions, seq_len, dim)
        attn = attention_mask.view(batch_size * num_expressions, seq_len)
        extended_mask = self.text_encoder.get_extended_attention_mask(attn, attn.shape, attn.device)
        for layer_idx in range(start_layer, end_layer):
            hidden = self.text_encoder.encoder.layer[layer_idx](hidden, extended_mask)[0]
        return hidden.view(batch_size, num_expressions, seq_len, dim)

    def _advance_visual(
        self,
        visual_state: Dict[str, int | torch.Tensor],
        target_block: int,
    ) -> torch.Tensor:
        x = visual_state["x"]
        while visual_state["global_block"] < target_block:
            stage_idx = visual_state["stage_idx"]
            block_idx = visual_state["block_idx"]
            stage = self.visual_encoder.layers[stage_idx]
            if block_idx == 0:
                batch_size, channels, depth, height, width = x.shape
                window_size, shift_size = get_window_size((depth, height, width), stage.window_size, stage.shift_size)
                depth_pad = int((depth + window_size[0] - 1) // window_size[0] * window_size[0])
                height_pad = int((height + window_size[1] - 1) // window_size[1] * window_size[1])
                width_pad = int((width + window_size[2] - 1) // window_size[2] * window_size[2])
                visual_state["attn_mask"] = compute_mask(depth_pad, height_pad, width_pad, window_size, shift_size, x.device)
                x = rearrange(x, "b c d h w -> b d h w c")
            x = stage.blocks[block_idx](x, visual_state["attn_mask"])
            visual_state["global_block"] += 1
            visual_state["block_idx"] += 1
            if visual_state["block_idx"] == len(stage.blocks):
                x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
                if stage.downsample is not None:
                    x = stage.downsample(x)
                x = rearrange(x, "b d h w c -> b c d h w")
                visual_state["x"] = x
                visual_state["stage_idx"] += 1
                visual_state["block_idx"] = 0
                visual_state["attn_mask"] = None
            else:
                visual_state["x"] = rearrange(x, "b d h w c -> b c d h w")
        return visual_state["x"]

    def _select_relevant_features(
        self,
        checkpoint_idx: int,
        visual_feature: torch.Tensor,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, depth, height, width = visual_feature.shape
        num_expressions = text_hidden.shape[1]
        visual_tokens = rearrange(visual_feature, "b c t h w -> b t (h w) c")
        text_to_visual = self.selection_text_to_visual[checkpoint_idx](text_hidden)
        sentence_embeddings = masked_mean(text_to_visual, text_mask, dim=2)

        visual_scores = F.cosine_similarity(
            visual_tokens.unsqueeze(2),
            sentence_embeddings.unsqueeze(1).unsqueeze(3),
            dim=-1,
        )
        kv = min(self.selection_topk_visual, visual_scores.shape[-1])
        topk_indices = visual_scores.topk(kv, dim=-1).indices
        proposal_mask = torch.zeros_like(visual_scores, dtype=torch.bool)
        proposal_mask.scatter_(-1, topk_indices, True)

        aggregated_visual_mask = proposal_mask.any(dim=2)
        visual_tokens = visual_tokens + visual_tokens * aggregated_visual_mask.unsqueeze(-1).float()

        vis_context = (
            visual_tokens.unsqueeze(2) * proposal_mask.unsqueeze(-1).float()
        ).sum(dim=3) / proposal_mask.float().sum(dim=3, keepdim=True).clamp_min(1.0)
        vis_context = vis_context.mean(dim=1)
        text_context = self.selection_visual_to_text[checkpoint_idx](vis_context)

        text_scores = F.cosine_similarity(text_hidden, text_context.unsqueeze(2), dim=-1)
        text_scores = text_scores.masked_fill(~text_mask.bool(), -1e4)
        kz = min(self.selection_topk_text, text_scores.shape[-1])
        text_topk = text_scores.topk(kz, dim=-1).indices
        text_mask_selected = torch.zeros_like(text_scores, dtype=torch.bool)
        text_mask_selected.scatter_(-1, text_topk, True)
        text_mask_selected &= text_mask.bool()
        text_hidden = text_hidden + text_hidden * text_mask_selected.unsqueeze(-1).float()

        visual_feature = rearrange(visual_tokens, "b t (h w) c -> b c t h w", h=height, w=width)
        return visual_feature, text_hidden

    def _interact(
        self,
        visual_feature: torch.Tensor,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, depth, height, width = visual_feature.shape
        num_expressions, seq_len = text_hidden.shape[1:3]
        video_tokens = rearrange(visual_feature, "b c t h w -> b (t h w) c")
        video_tokens = self.video_proj(video_tokens)

        positive_tokens = self.text_proj(text_hidden[:, : self.num_positive])
        positive_tokens = positive_tokens.reshape(batch_size, -1, self.hidden_dim)
        positive_mask = ~text_mask[:, : self.num_positive].reshape(batch_size, -1).bool()
        video_tokens = self.video_interaction(video_tokens, positive_tokens, positive_mask)

        text_tokens = self.text_proj(text_hidden).reshape(batch_size * num_expressions, seq_len, self.hidden_dim)
        repeated_video = video_tokens.unsqueeze(1).expand(-1, num_expressions, -1, -1).reshape(
            batch_size * num_expressions,
            video_tokens.shape[1],
            self.hidden_dim,
        )
        text_tokens = self.text_interaction(
            text_tokens,
            repeated_video,
            None,
        )
        text_tokens = text_tokens.view(batch_size, num_expressions, seq_len, self.hidden_dim)
        expression_embeddings = masked_mean(text_tokens, text_mask, dim=2)

        video_tokens = rearrange(video_tokens, "b (t n) c -> b t n c", t=depth, n=height * width)
        return video_tokens, expression_embeddings

    def _mil_scores(
        self,
        video_tokens: torch.Tensor,
        expression_embeddings: torch.Tensor,
        valid_expression_mask: torch.Tensor,
    ) -> torch.Tensor:
        similarity = F.cosine_similarity(
            video_tokens.unsqueeze(2),
            expression_embeddings.unsqueeze(1).unsqueeze(3),
            dim=-1,
        ).permute(0, 1, 3, 2)
        proposal_mask = similarity.sigmoid() > self.mask_threshold
        proposal_sum = (
            video_tokens.unsqueeze(3) * proposal_mask.unsqueeze(-1).float()
        ).sum(dim=2)
        proposal_den = proposal_mask.float().sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        proposal_features = proposal_sum / proposal_den

        u_cls = torch.einsum("btec,bkc->btek", proposal_features, expression_embeddings)
        u_cls = u_cls.masked_fill(~valid_expression_mask[:, None, None, :], -1e4)
        u_cls = u_cls.softmax(dim=-1)

        u_smt = self.segmentation_flow(proposal_features)[..., : expression_embeddings.shape[1]]
        u_smt = u_smt.masked_fill(~valid_expression_mask[:, None, None, :], -1e4)
        u_smt = u_smt.softmax(dim=2)
        return (u_cls * u_smt).sum(dim=2), similarity

    def forward(self, batch: Dict[str, List | torch.Tensor]) -> Dict[str, torch.Tensor]:
        videos = batch["videos"].tensors.permute(1, 0, 2, 3, 4).contiguous()
        device = videos.device
        batch_size = videos.shape[0]

        expression_texts, labels, valid_expression_mask, positive_confidences = self._prepare_expressions(
            batch["original_texts"],
            batch["positive_texts"],
            batch["negative_texts"],
            batch["positive_confidences"],
            device,
        )
        text_hidden, text_mask = self._tokenize(expression_texts, batch_size, self.max_expressions, device)

        visual_feature = videos.permute(0, 2, 1, 3, 4)
        visual_feature = self.visual_encoder.patch_embed(visual_feature)
        visual_feature = self.visual_encoder.pos_drop(visual_feature)
        visual_state = {
            "x": visual_feature,
            "stage_idx": 0,
            "block_idx": 0,
            "global_block": 0,
            "attn_mask": None,
        }

        start_text_layer = 0
        for checkpoint_pos, target_layer in enumerate(self.selection_checkpoints):
            visual_feature = self._advance_visual(visual_state, target_layer)
            text_hidden = self._advance_text(text_hidden, text_mask, start_text_layer, target_layer)
            start_text_layer = target_layer
            visual_feature, text_hidden = self._select_relevant_features(
                checkpoint_pos,
                visual_feature,
                text_hidden,
                text_mask,
            )
            if visual_state["block_idx"] != 0:
                visual_state["x"] = rearrange(visual_feature, "b c t h w -> b t h w c")
            else:
                visual_state["x"] = visual_feature

        visual_feature = rearrange(visual_feature, "b c t h w -> b t h w c")
        visual_feature = self.visual_encoder.norm(visual_feature)
        visual_feature = rearrange(visual_feature, "b t h w c -> b c t h w")

        video_tokens, expression_embeddings = self._interact(visual_feature, text_hidden, text_mask)
        frame_scores, similarity = self._mil_scores(video_tokens, expression_embeddings, valid_expression_mask)

        positive_similarity = similarity[..., : self.num_positive]
        confidence_weights = positive_confidences / positive_confidences.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        fused_similarity = (positive_similarity * confidence_weights[:, None, None, :]).sum(dim=-1)
        pseudo_masks = fused_similarity.sigmoid() > self.mask_threshold

        loss_cls = classification_loss(frame_scores, labels, valid_expression_mask)
        loss_focal, loss_dice = segmentation_loss(positive_similarity, pseudo_masks.float())
        loss_tmp = temporal_ranking_loss(pseudo_masks.float(), self.ranking_epsilon)
        total_loss = loss_cls + self.lambda_seg * (loss_focal + loss_dice) + self.lambda_tmp * loss_tmp

        height, width = visual_feature.shape[-2:]
        pred_masks = fused_similarity.view(batch_size, fused_similarity.shape[1], height, width)

        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_focal": loss_focal.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_tmp": loss_tmp.detach(),
            "frame_scores": frame_scores.detach(),
            "similarity": similarity.detach(),
            "pred_masks": pred_masks,
            "pseudo_masks": pseudo_masks.float().view(batch_size, pseudo_masks.shape[1], height, width),
        }

    @torch.no_grad()
    def infer(self, videos, text_queries: List[str]) -> torch.Tensor:
        dummy_batch = {
            "videos": videos,
            "original_texts": text_queries,
            "positive_texts": [[text] for text in text_queries],
            "positive_confidences": [[1.0] for _ in text_queries],
            "negative_texts": [[] for _ in text_queries],
        }
        outputs = self.forward(dummy_batch)
        return outputs["pred_masks"].sigmoid()
