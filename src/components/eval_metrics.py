from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall
## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        # TODO (Clark): in the future we might need to have non-collision negative sampling
        # when K in top-k is very small (e.g., hits@1) and num_negatives is very large
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, we get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # we initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # we set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )


class BehaviorAccuracy(CustomMeanReductionMetric):
    """Tracks per-batch behavior prediction accuracy (correct / total)."""

    def update(self, correct: int, total: int) -> None:
        self.metric_values += correct
        self.total_values += total


class BehaviorClassMetric(torchmetrics.Metric):
    """
    Per-behavior precision / recall / F1 for the predicted behavior class.

    Accumulates one-vs-rest TP / FP / FN counts for a single behavior id and
    reduces them across workers in ``compute`` (so ``sync_on_compute=False``).
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = TP / (TP + 0.5 * (FP + FN))   ==  2TP / (2TP + FP + FN)
    A zero denominator yields 0.0 (convention for an undefined class metric).
    """

    def __init__(self, behavior_id: int, metric_type: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if metric_type not in ("precision", "recall", "f1"):
            raise ValueError(f"Unknown metric_type: {metric_type}")
        self.behavior_id = behavior_id
        self.metric_type = metric_type
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred_behavior: torch.Tensor, gt_behavior: torch.Tensor) -> None:
        pred_c = pred_behavior == self.behavior_id
        gt_c = gt_behavior == self.behavior_id
        self.tp += int((pred_c & gt_c).sum().item())
        self.fp += int((pred_c & ~gt_c).sum().item())
        self.fn += int((~pred_c & gt_c).sum().item())

    def compute(self) -> torch.Tensor:
        tp = torch.tensor(float(self.tp), device=self.device)
        fp = torch.tensor(float(self.fp), device=self.device)
        fn = torch.tensor(float(self.fn), device=self.device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            def _gather_sum(t: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    [g.unsqueeze(0) if g.dim() == 0 else g for g in gather_all_tensors(t)]
                ).sum()

            tp, fp, fn = _gather_sum(tp), _gather_sum(fp), _gather_sum(fn)

        if self.metric_type == "precision":
            denom = tp + fp
        elif self.metric_type == "recall":
            denom = tp + fn
        else:  # f1
            denom = tp + 0.5 * (fp + fn)

        if denom.item() == 0:
            return torch.tensor(0.0, device=self.device)
        return tp / denom

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0


class MultiBehaviorSIDRetrievalEvaluator(Evaluator):
    """
    Evaluator for multi-behavior TIGER.  Expects generated_ids of shape
    (B, top_k, 1+H) where [:,:,0] is the predicted behavior and [:,:,1:]
    are the predicted SID levels.  Labels shape: (B, 1+H).

    Metric names use underscores (safe for nn.Module registration):
        t1_ndcg_K, t1_recall_K  — GT behavior == buy → SID HR/NDCG
        t2_ndcg_K, t2_recall_K  — conditioned on GT behavior → SID HR/NDCG
        t3_ndcg_K, t3_recall_K  — free generation → joint (behavior+SID) HR/NDCG
        t3_behavior_accuracy     — free generation → behavior prediction accuracy
        t3_behavior_{precision,recall,f1}_{pv,fav,cart,buy}
                                 — free generation → per-behavior classification metrics
                                   (best-beam predicted behavior vs GT behavior, one-vs-rest)
    """

    # Default behavior id -> short name used in per-behavior metric names.
    DEFAULT_BEHAVIOR_NAMES = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        buy_behavior_id: int = 3,
        behavior_names: Dict[int, str] = None,
    ):
        def _make(prefix, name, obj):
            return {
                f"{prefix}_{name}_{k}": obj(top_k=k, sync_on_compute=False, compute_with_cache=False)
                for k in top_k_list
            }

        all_metrics: Dict[str, Any] = {}
        for track in ("t1", "t2", "t3"):
            all_metrics.update(_make(track, "ndcg", metrics["ndcg"]))
            all_metrics.update(_make(track, "recall", metrics["recall"]))
        all_metrics["t3_behavior_accuracy"] = BehaviorAccuracy(
            sync_on_compute=False, compute_with_cache=False
        )

        # Per-behavior precision / recall / f1 on the free-generation (t3) behavior prediction.
        self.behavior_names = (
            behavior_names if behavior_names is not None else self.DEFAULT_BEHAVIOR_NAMES
        )
        for bid, bname in self.behavior_names.items():
            for mtype in ("precision", "recall", "f1"):
                all_metrics[f"t3_behavior_{mtype}_{bname}"] = BehaviorClassMetric(
                    behavior_id=bid,
                    metric_type=mtype,
                    sync_on_compute=False,
                    compute_with_cache=False,
                )

        self.buy_behavior_id = buy_behavior_id
        self.metrics = all_metrics

    def _update_retrieval(
        self,
        track: str,
        marginal_probs: torch.Tensor,
        generated_sids: torch.Tensor,
        gt_sids: torch.Tensor,
    ):
        """
        Args:
            track:           "t1", "t2", or "t3"
            marginal_probs:  (B, top_k)
            generated_sids:  (B, top_k, H)  — SID columns only
            gt_sids:         (B, H)
        """
        batch_size, num_candidates, num_h = generated_sids.shape
        labels = gt_sids.reshape(batch_size, 1, num_h)
        preds = marginal_probs.reshape(-1)

        matched = torch.all((generated_sids == labels), dim=2).nonzero()
        target = torch.zeros(batch_size, num_candidates, dtype=torch.bool,
                             device=generated_sids.device)
        if matched.numel() > 0:
            target[matched[:, 0], matched[:, 1]] = True
        target = target.reshape(-1)

        expanded_indexes = (
            torch.arange(batch_size, device=generated_sids.device)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for key, metric_object in self.metrics.items():
            if key.startswith(track) and isinstance(metric_object, CustomRetrievalMetric):
                metric_object.update(
                    preds,
                    target.to(preds.device),
                    indexes=expanded_indexes.to(preds.device),
                )

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        marginal_probs_t2: torch.Tensor,
        generated_ids_t2: torch.Tensor,
    ):
        """
        Args:
            marginal_probs:    (B, top_k)
            generated_ids:     (B, top_k, 1+H)
            labels:            (B, 1+H)  labels[:,0]=GT behavior, labels[:,1:]=GT SIDs
            marginal_probs_t2: (B, top_k)
            generated_ids_t2:  (B, top_k, H)  SIDs only (behavior conditioned)
        """
        batch_size = labels.size(0)
        gt_behavior = labels[:, 0]        # (B,)
        gt_sids = labels[:, 1:]           # (B, H)

        pred_behavior = generated_ids[:, :, 0]   # (B, top_k)
        pred_sids = generated_ids[:, :, 1:]      # (B, top_k, H)

        # ---- Track 1: GT behavior == buy ----
        buy_mask = (gt_behavior == self.buy_behavior_id)
        if buy_mask.any():
            self._update_retrieval("t1", marginal_probs[buy_mask], pred_sids[buy_mask], gt_sids[buy_mask])

        # ---- Track 2: conditioned on GT behavior ----
        self._update_retrieval("t2", marginal_probs_t2, generated_ids_t2, gt_sids)

        # ---- Track 3: joint behavior+SID HR/NDCG ----
        behavior_match = (pred_behavior == gt_behavior.unsqueeze(1))   # (B, top_k)
        sid_match = torch.all((pred_sids == gt_sids.unsqueeze(1)), dim=2)  # (B, top_k)
        joint_match = (behavior_match & sid_match).reshape(-1)
        preds_flat = marginal_probs.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size, device=labels.device)
            .unsqueeze(-1)
            .expand(batch_size, generated_ids.size(1))
            .reshape(-1)
        )
        for key, metric_object in self.metrics.items():
            if key.startswith("t3") and isinstance(metric_object, CustomRetrievalMetric):
                metric_object.update(
                    preds_flat,
                    joint_match.to(preds_flat.device),
                    indexes=expanded_indexes.to(preds_flat.device),
                )

        # ---- Track 3: behavior accuracy + per-behavior P/R/F1 (best beam) ----
        best_beam = marginal_probs.argmax(dim=-1)  # (B,)
        best_pred = pred_behavior[torch.arange(batch_size, device=pred_behavior.device), best_beam]
        correct = int((best_pred == gt_behavior).sum().item())
        self.metrics["t3_behavior_accuracy"].update(correct, batch_size)

        # Per-behavior precision / recall / f1 over the best-beam predicted behavior.
        for metric_object in self.metrics.values():
            if isinstance(metric_object, BehaviorClassMetric):
                metric_object.update(best_pred, gt_behavior)
