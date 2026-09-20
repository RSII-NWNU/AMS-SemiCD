import torch
from torch import Tensor


class FreeMatchStyleHook:
    """Pixel-level adaptation of FreeMatch self-adaptive thresholding."""

    def __init__(self, num_classes=2, momentum=0.999, eps=1e-12):
        if num_classes != 2:
            raise ValueError("FreeMatchStyleHook currently supports binary change detection only")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.p_model = torch.full((self.num_classes,), 1.0 / self.num_classes)
        self.label_hist = torch.full((self.num_classes,), 1.0 / self.num_classes)
        self.time_p = torch.tensor(1.0 / self.num_classes)
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
        self.p_model = self.p_model.to(device=ref.device, dtype=ref.dtype)
        self.label_hist = self.label_hist.to(device=ref.device, dtype=ref.dtype)
        self.time_p = self.time_p.to(device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def _update(self, probs: Tensor, confidence: Tensor, pred_class: Tensor):
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        batch_p_model = flat_probs.mean(dim=0)
        hist = torch.bincount(pred_class.reshape(-1), minlength=self.num_classes).to(probs.dtype)
        hist = hist / hist.sum().clamp_min(1.0)
        one_minus_m = 1.0 - self.momentum
        self.time_p.mul_(self.momentum).add_(confidence.mean(), alpha=one_minus_m)
        self.p_model.mul_(self.momentum).add_(batch_p_model, alpha=one_minus_m)
        self.label_hist.mul_(self.momentum).add_(hist, alpha=one_minus_m)

    @torch.no_grad()
    def _thresholds(self) -> Tensor:
        modulation = self.p_model / self.p_model.max().clamp_min(self.eps)
        return self.time_p * modulation

    @torch.no_grad()
    def masking(
        self, algorithm, epoch, logits_x_ulb, is_logits=True, is_update=True, total_epochs=None
    ) -> Tensor:
        del algorithm, total_epochs
        probs = self._to_probs(logits_x_ulb.detach(), is_logits)
        confidence, pred_class = probs.max(dim=1)
        self._ensure_device(confidence)
        if is_update:
            self._update(probs, confidence, pred_class)
        thresholds = self._thresholds()
        mask = confidence.ge(thresholds[pred_class]).to(confidence.dtype)
        self._last_monitor = {
            "epoch": int(epoch),
            "mean_weight": float(mask.mean().item()),
            "selected_ratio": float(mask.mean().item()),
            "unch_threshold": float(thresholds[0].item()),
            "ch_threshold": float(thresholds[1].item()),
            "time_p": float(self.time_p.item()),
        }
        return mask

    @torch.no_grad()
    def get_thresholds(self, algorithm=None, epoch=None):
        del algorithm, epoch
        thresholds = self._thresholds()
        sigma = torch.zeros_like(thresholds)
        return thresholds, sigma, self.p_model.clone(), self.label_hist.clone()

    def get_monitor_metrics(self, algorithm=None):
        del algorithm
        return dict(self._last_monitor)

    def state_dict(self):
        return {
            "p_model": self.p_model.detach().cpu().clone(),
            "label_hist": self.label_hist.detach().cpu().clone(),
            "time_p": self.time_p.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        self.p_model = state_dict["p_model"].detach().clone().to(torch.float32)
        self.label_hist = state_dict["label_hist"].detach().clone().to(torch.float32)
        self.time_p = state_dict["time_p"].detach().clone().to(torch.float32)

