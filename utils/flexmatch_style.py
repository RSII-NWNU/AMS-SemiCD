import torch
from torch import Tensor


class FlexMatchStyleHook:
    """Pixel-level adaptation of FlexMatch curriculum thresholding."""

    def __init__(self, num_classes=2, p_cutoff=0.95, eps=1e-12):
        if num_classes != 2:
            raise ValueError("FlexMatchStyleHook currently supports binary change detection only")
        self.num_classes = int(num_classes)
        self.p_cutoff = float(p_cutoff)
        self.eps = float(eps)
        self.selected_count = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unselected_count = torch.tensor(0.0, dtype=torch.float64)
        self.classwise_acc = torch.zeros(self.num_classes, dtype=torch.float32)
        self._last_monitor = {}

    @torch.no_grad()
    def _to_probs(self, values: Tensor, is_logits: bool) -> Tensor:
        if values.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(values.shape)}")
        if values.shape[1] == 1:
            change_prob = torch.sigmoid(values) if is_logits else values
            change_prob = change_prob.clamp(0.0, 1.0)
            return torch.cat((1.0 - change_prob, change_prob), dim=1)
        if values.shape[1] != self.num_classes:
            raise ValueError(f"Expected 1 or {self.num_classes} channels, got {values.shape[1]}")
        return torch.softmax(values, dim=1) if is_logits else values

    @torch.no_grad()
    def _ensure_device(self, ref: Tensor):
        self.selected_count = self.selected_count.to(ref.device)
        self.unselected_count = self.unselected_count.to(ref.device)
        self.classwise_acc = self.classwise_acc.to(device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def _thresholds(self, ref: Tensor) -> Tensor:
        self._ensure_device(ref)
        return self.p_cutoff * self.classwise_acc / (2.0 - self.classwise_acc).clamp_min(self.eps)

    @torch.no_grad()
    def _update(self, confidence: Tensor, pred_class: Tensor):
        base_selected = confidence.ge(self.p_cutoff)
        if base_selected.any():
            counts = torch.bincount(
                pred_class[base_selected].reshape(-1), minlength=self.num_classes
            ).to(self.selected_count.dtype)
            self.selected_count.add_(counts)
        self.unselected_count.add_((~base_selected).sum().to(self.unselected_count.dtype))

        denominator = torch.maximum(self.selected_count.max(), self.unselected_count).clamp_min(1.0)
        self.classwise_acc.copy_((self.selected_count / denominator).to(self.classwise_acc.dtype))

    @torch.no_grad()
    def masking(
        self, algorithm, epoch, logits_x_ulb, is_logits=True, is_update=True, total_epochs=None
    ) -> Tensor:
        del algorithm, total_epochs
        probs = self._to_probs(logits_x_ulb.detach(), is_logits)
        confidence, pred_class = probs.max(dim=1)
        thresholds = self._thresholds(confidence)
        mask = confidence.ge(thresholds[pred_class]).to(confidence.dtype)

        if is_update:
            self._update(confidence, pred_class)

        current_thresholds = self._thresholds(confidence)
        self._last_monitor = {
            "epoch": int(epoch),
            "mean_weight": float(mask.mean().item()),
            "selected_ratio": float(mask.mean().item()),
            "unch_threshold": float(current_thresholds[0].item()),
            "ch_threshold": float(current_thresholds[1].item()),
        }
        return mask

    @torch.no_grad()
    def get_thresholds(self, algorithm=None, epoch=None):
        del algorithm, epoch
        thresholds = self.p_cutoff * self.classwise_acc / (
            2.0 - self.classwise_acc
        ).clamp_min(self.eps)
        zeros = torch.zeros_like(thresholds)
        return thresholds, zeros, self.classwise_acc.clone(), zeros

    def get_monitor_metrics(self, algorithm=None):
        del algorithm
        return dict(self._last_monitor)

    def state_dict(self):
        return {
            "selected_count": self.selected_count.detach().cpu().clone(),
            "unselected_count": self.unselected_count.detach().cpu().clone(),
            "classwise_acc": self.classwise_acc.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        self.selected_count = state_dict["selected_count"].detach().clone().to(torch.float64)
        self.unselected_count = state_dict["unselected_count"].detach().clone().to(torch.float64)
        self.classwise_acc = state_dict["classwise_acc"].detach().clone().to(torch.float32)

