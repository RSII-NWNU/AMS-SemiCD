import torch
from torch import Tensor


class SoftMatchStyleHook:
    """Pixel-level SoftMatch truncated Gaussian weighting."""

    def __init__(self, num_classes=2, n_sigma=2.0, momentum=0.999, eps=1e-12):
        if num_classes != 2:
            raise ValueError("SoftMatchStyleHook currently supports binary change detection only")
        self.num_classes = int(num_classes)
        self.n_sigma = float(n_sigma)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.prob_max_mu = torch.tensor(1.0 / self.num_classes)
        self.prob_max_var = torch.tensor(1.0)
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
        self.prob_max_mu = self.prob_max_mu.to(device=ref.device, dtype=ref.dtype)
        self.prob_max_var = self.prob_max_var.to(device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def _update(self, confidence: Tensor):
        batch_mu = confidence.mean()
        batch_var = confidence.var(unbiased=True) if confidence.numel() > 1 else self.prob_max_var
        one_minus_m = 1.0 - self.momentum
        self.prob_max_mu.mul_(self.momentum).add_(batch_mu, alpha=one_minus_m)
        self.prob_max_var.mul_(self.momentum).add_(batch_var, alpha=one_minus_m)

    @torch.no_grad()
    def masking(
        self, algorithm, epoch, logits_x_ulb, is_logits=True, is_update=True, total_epochs=None
    ) -> Tensor:
        del algorithm, total_epochs
        probs = self._to_probs(logits_x_ulb.detach(), is_logits)
        confidence = probs.max(dim=1).values
        self._ensure_device(confidence)
        if is_update:
            self._update(confidence)

        left_difference = torch.clamp(confidence - self.prob_max_mu, max=0.0)
        denominator = 2.0 * self.prob_max_var.clamp_min(self.eps) / (self.n_sigma ** 2)
        weights = torch.exp(-(left_difference.square() / denominator))
        self._last_monitor = {
            "epoch": int(epoch),
            "mean_weight": float(weights.mean().item()),
            "selected_ratio": float(weights.gt(0.5).float().mean().item()),
            "mu": float(self.prob_max_mu.item()),
            "var": float(self.prob_max_var.item()),
        }
        return weights

    @torch.no_grad()
    def get_thresholds(self, algorithm=None, epoch=None):
        del algorithm, epoch
        threshold = self.prob_max_mu.reshape(1).repeat(self.num_classes)
        sigma = torch.full_like(threshold, self.n_sigma)
        mu = self.prob_max_mu.reshape(1).repeat(self.num_classes)
        var = self.prob_max_var.reshape(1).repeat(self.num_classes)
        return threshold, sigma, mu, var

    def get_monitor_metrics(self, algorithm=None):
        del algorithm
        return dict(self._last_monitor)

    def state_dict(self):
        return {
            "prob_max_mu": self.prob_max_mu.detach().cpu().clone(),
            "prob_max_var": self.prob_max_var.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        self.prob_max_mu = state_dict["prob_max_mu"].detach().clone().to(torch.float32)
        self.prob_max_var = state_dict["prob_max_var"].detach().clone().to(torch.float32)

