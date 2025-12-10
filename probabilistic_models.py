"""Probabilistic FIN with Input-Dependent Uncertainty using Beta Distribution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureNormalizer(nn.Module):
    """Normalizes features to [0,1] range for Beta distribution."""

    def __init__(self, feature_dim=256, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.feature_dim = feature_dim

        self.register_buffer('running_min', torch.zeros(feature_dim))
        self.register_buffer('running_max', torch.ones(feature_dim))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def update_stats(self, features):
        if not self.training:
            return

        with torch.no_grad():
            batch_min = features.min(dim=0)[0]
            batch_max = features.max(dim=0)[0]

            if not self.initialized:
                self.running_min.copy_(batch_min)
                self.running_max.copy_(batch_max)
                self.initialized.fill_(True)
            else:
                self.running_min.copy_((1 - self.momentum) * self.running_min + self.momentum * batch_min)
                self.running_max.copy_((1 - self.momentum) * self.running_max + self.momentum * batch_max)

            self.num_batches_tracked.add_(1)

    def normalize(self, features):
        if not self.initialized:
            feat_min = features.min(dim=0, keepdim=True)[0]
            feat_max = features.max(dim=0, keepdim=True)[0]
        else:
            feat_min = self.running_min.unsqueeze(0)
            feat_max = self.running_max.unsqueeze(0)

        range_val = feat_max - feat_min + self.eps
        normalized = (features - feat_min) / range_val
        return torch.clamp(normalized, self.eps, 1 - self.eps)

    def denormalize(self, normalized_features):
        if not self.initialized:
            return normalized_features

        feat_min = self.running_min.unsqueeze(0)
        feat_max = self.running_max.unsqueeze(0)
        range_val = feat_max - feat_min + self.eps
        return normalized_features * range_val + feat_min


class BetaNLLLoss(nn.Module):
    """
    Beta-NLL Loss from Seitzer et al., 2022.
    Variance in weighting term is detached to prevent trivial minimization.
    """

    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, mean, variance, target):
        variance = torch.clamp(variance, min=1e-6)
        nll = 0.5 * ((target - mean) ** 2 / variance + variance.log())

        if self.beta > 0:
            nll = nll * (variance.detach() ** self.beta)

        return nll.sum(dim=-1).mean()

    def forward_with_components(self, mean, variance, target):
        variance = torch.clamp(variance, min=1e-6)

        sq_error = (target - mean) ** 2
        mse_term = 0.5 * sq_error / variance
        log_var_term = 0.5 * variance.log()
        nll = mse_term + log_var_term

        if self.beta > 0:
            weight = variance.detach() ** self.beta
            weighted_nll = nll * weight
        else:
            weighted_nll = nll
            weight = torch.ones_like(variance)

        return {
            'mse_term': mse_term.mean().item(),
            'log_var_term': log_var_term.mean().item(),
            'nll': nll.mean().item(),
            'beta_weight': weight.mean().item(),
            'weighted_nll': weighted_nll.mean().item(),
            'variance': variance.mean().item(),
            'total': weighted_nll.sum(dim=-1).mean().item()
        }


class InputDependentUncertaintyHead(nn.Module):
    """Produces input-dependent alpha/beta parameters for Beta distribution."""

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.alpha_head, self.beta_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.constant_(self.alpha_head[-1].bias, 1.0)
        nn.init.constant_(self.beta_head[-1].bias, 1.0)

    def forward(self, transformer_output, input_features):
        input_cond = self.input_proj(input_features)
        combined = torch.cat([transformer_output, input_cond], dim=-1)

        alpha = F.softplus(self.alpha_head(combined)) + 1.01
        beta = F.softplus(self.beta_head(combined)) + 1.01

        alpha = torch.clamp(alpha, 1.01, 100.0)
        beta = torch.clamp(beta, 1.01, 100.0)

        return alpha, beta


class ProbabilisticFIN(nn.Module):
    """
    Probabilistic Feature Imputation Network with input-dependent uncertainty.
    Uses Beta distribution for bounded features with L2-normalized output.
    """

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=256,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.uncertainty_head = InputDependentUncertaintyHead(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )

        self.normalizer = FeatureNormalizer(feature_dim=output_dim)

    def forward(self, x, target_features=None, return_normalized=True,
                return_samples=False, num_samples=1):
        batch_size = x.shape[0]
        original_input = x

        if target_features is not None and self.training:
            self.normalizer.update_stats(target_features)

        x_proj = self.input_proj(x).unsqueeze(1)
        query = self.query_token.expand(batch_size, -1, -1)

        sequence = torch.cat([query, x_proj], dim=1)
        sequence = sequence + self.pos_embedding

        encoded = self.transformer(sequence)
        transformer_output = encoded[:, 0, :]

        alpha, beta = self.uncertainty_head(transformer_output, original_input)
        mean = alpha / (alpha + beta)

        if return_normalized:
            output_features = mean
        else:
            denorm_features = self.normalizer.denormalize(mean)
            output_features = F.normalize(denorm_features, p=2, dim=-1)

        if return_samples:
            dist = torch.distributions.Beta(alpha, beta)
            samples = dist.rsample((num_samples,))
            return mean, alpha, beta, samples

        return alpha, beta, output_features

    def get_uncertainty(self, alpha, beta):
        """Compute variance of Beta distribution: alpha*beta / ((a+b)^2 * (a+b+1))"""
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return variance

    def get_confidence(self, alpha, beta):
        variance = self.get_uncertainty(alpha, beta)
        confidence = 1.0 / (variance + 1e-6)
        return confidence

    def impute(self, x, use_mean=True):
        _, _, features = self.forward(x, return_normalized=False)
        return features

    def impute_with_uncertainty(self, x):
        alpha, beta, features = self.forward(x, return_normalized=False)
        uncertainty = self.get_uncertainty(alpha, beta)
        mean_uncertainty = uncertainty.mean(dim=-1)
        return features, uncertainty, mean_uncertainty


class DeterministicFIN(nn.Module):
    """Deterministic FIN baseline with L2 normalization."""

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=256,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x_proj = self.input_proj(x).unsqueeze(1)
        query = self.query_token.expand(batch_size, -1, -1)
        sequence = torch.cat([query, x_proj], dim=1)
        sequence = sequence + self.pos_embedding

        encoded = self.transformer(sequence)
        output = self.output_proj(encoded[:, 0, :])
        output = F.normalize(output, p=2, dim=-1)

        return output

    def impute(self, x):
        return self.forward(x)


def normalize_features(features, eps=1e-5):
    """Normalize features to [0,1] range using batch statistics."""
    feat_min = features.min(dim=0, keepdim=True)[0]
    feat_max = features.max(dim=0, keepdim=True)[0]
    normalized = (features - feat_min) / (feat_max - feat_min + eps)
    return torch.clamp(normalized, eps, 1 - eps)


def test_input_dependent_uncertainty():
    print("Testing input-dependent uncertainty...")

    model = ProbabilisticFIN(input_dim=256, output_dim=256)
    model.eval()

    torch.manual_seed(42)
    x1 = torch.randn(1, 256)
    x2 = torch.randn(1, 256) * 2
    x3 = torch.zeros(1, 256)
    x4 = torch.ones(1, 256)

    with torch.no_grad():
        alpha1, beta1, _ = model(x1)
        alpha2, beta2, _ = model(x2)
        alpha3, beta3, _ = model(x3)
        alpha4, beta4, _ = model(x4)

        var1 = model.get_uncertainty(alpha1, beta1).mean().item()
        var2 = model.get_uncertainty(alpha2, beta2).mean().item()
        var3 = model.get_uncertainty(alpha3, beta3).mean().item()
        var4 = model.get_uncertainty(alpha4, beta4).mean().item()

    print(f"  Input 1 (random):  uncertainty = {var1:.6f}")
    print(f"  Input 2 (scaled):  uncertainty = {var2:.6f}")
    print(f"  Input 3 (zeros):   uncertainty = {var3:.6f}")
    print(f"  Input 4 (ones):    uncertainty = {var4:.6f}")

    uncertainties = [var1, var2, var3, var4]
    if len(set([f"{u:.6f}" for u in uncertainties])) > 1:
        print("SUCCESS: Uncertainties vary across inputs!")
    else:
        print("WARNING: Uncertainties are still constant!")

    batch_inputs = torch.randn(32, 256)
    with torch.no_grad():
        alpha_batch, beta_batch, _ = model(batch_inputs)
        var_batch = model.get_uncertainty(alpha_batch, beta_batch).mean(dim=1)

    print(f"\n  Batch variance of uncertainty: {var_batch.std().item():.6f}")
    print(f"  Min uncertainty in batch: {var_batch.min().item():.6f}")
    print(f"  Max uncertainty in batch: {var_batch.max().item():.6f}")

    print("\nTesting output normalization...")
    with torch.no_grad():
        _, _, features = model(batch_inputs, return_normalized=False)
        norms = torch.norm(features, p=2, dim=-1)
        print(f"  Output norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.4f}")

    return True


if __name__ == "__main__":
    test_input_dependent_uncertainty()
