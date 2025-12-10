"""Federated Learning Framework with Uncertainty-Weighted Aggregation."""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Tuple, Optional
import numpy as np
import math


def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: Optional[List[float]] = None
) -> nn.Module:
    """Standard FedAvg aggregation."""
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    global_dict = global_model.state_dict()
    device = next(iter(global_dict.values())).device

    for key in global_dict.keys():
        if not global_dict[key].dtype.is_floating_point:
            global_dict[key] = client_models[0].state_dict()[key].clone().to(device)
            continue

        global_dict[key] = torch.zeros_like(global_dict[key])
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            global_dict[key] += weight * client_dict[key].to(device)

    global_model.load_state_dict(global_dict)
    return global_model


def uq_weighted_fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_uncertainties: List[float],
    client_data_sizes: List[int],
    multimodal_indices: Optional[List[int]] = None,
    alpha: float = 0.3,
    temperature: float = 0.5,
    multimodal_boost: float = 3.0,
    min_weight: float = 0.01
) -> nn.Module:
    """Uncertainty-weighted FedAvg with multimodal client boosting."""
    total_data = sum(client_data_sizes)
    data_weights = [n / total_data for n in client_data_sizes]

    if all(u == 0 for u in client_uncertainties):
        return fedavg_aggregate(global_model, client_models, data_weights)

    scaled_confidences = []
    for unc in client_uncertainties:
        unc = max(unc, 1e-8)
        conf = math.exp(-unc / temperature)
        scaled_confidences.append(conf)

    total_conf = sum(scaled_confidences)
    norm_confidences = [c / total_conf for c in scaled_confidences]

    client_weights = []
    for i, (dw, conf) in enumerate(zip(data_weights, norm_confidences)):
        weight = (1 - alpha) * dw + alpha * conf

        if multimodal_indices is not None and i in multimodal_indices:
            weight *= multimodal_boost

        weight = max(weight, min_weight)
        client_weights.append(weight)

    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    print(f"\nClient weights (alpha={alpha}, temp={temperature}, mm_boost={multimodal_boost}):")
    for i, (dw, conf, w, unc) in enumerate(
        zip(data_weights, norm_confidences, client_weights, client_uncertainties)
    ):
        is_mm = multimodal_indices is not None and i in multimodal_indices
        mm_tag = " [MM]" if is_mm else ""
        print(f"  Client {i}{mm_tag}: data_wt={dw:.3f}, conf={conf:.3f}, "
              f"final_wt={w:.3f}, unc={unc:.6f}")

    return fedavg_aggregate(global_model, client_models, client_weights)


def aggregate_fins(
    global_fin_i2t: nn.Module,
    global_fin_t2i: nn.Module,
    client_fins_i2t: List[nn.Module],
    client_fins_t2i: List[nn.Module],
    multimodal_indices: List[int],
    client_uncertainties: Optional[List[float]] = None,
    use_uncertainty_weighting: bool = False
) -> Tuple[nn.Module, nn.Module]:
    """Aggregate FIN models from multimodal clients only."""
    if len(multimodal_indices) == 0:
        return global_fin_i2t, global_fin_t2i

    mm_fins_i2t = [client_fins_i2t[i] for i in multimodal_indices]
    mm_fins_t2i = [client_fins_t2i[i] for i in multimodal_indices]

    num_mm_clients = len(multimodal_indices)

    if use_uncertainty_weighting and client_uncertainties is not None:
        mm_uncertainties = [client_uncertainties[i] for i in multimodal_indices]

        if all(u == 0 for u in mm_uncertainties):
            weights = [1.0 / num_mm_clients] * num_mm_clients
        else:
            confidences = [1.0 / (u + 1e-6) for u in mm_uncertainties]
            total_conf = sum(confidences)
            weights = [c / total_conf for c in confidences]
    else:
        weights = [1.0 / num_mm_clients] * num_mm_clients

    global_fin_i2t = fedavg_aggregate(global_fin_i2t, mm_fins_i2t, weights)
    global_fin_t2i = fedavg_aggregate(global_fin_t2i, mm_fins_t2i, weights)

    return global_fin_i2t, global_fin_t2i


class FederatedTrainer:
    """Handles federated training with uncertainty-weighted aggregation."""

    def __init__(
        self,
        global_model: nn.Module,
        global_fin_i2t: nn.Module,
        global_fin_t2i: nn.Module,
        num_clients: int,
        use_uq_weighting: bool = True,
        uq_alpha: float = 0.3,
        uq_temperature: float = 0.5,
        warmup_rounds: int = 5,
        multimodal_boost: float = 3.0
    ):
        self.global_model = global_model
        self.global_fin_i2t = global_fin_i2t
        self.global_fin_t2i = global_fin_t2i
        self.num_clients = num_clients
        self.use_uq_weighting = use_uq_weighting
        self.uq_alpha = uq_alpha
        self.uq_temperature = uq_temperature
        self.warmup_rounds = warmup_rounds
        self.multimodal_boost = multimodal_boost
        self.current_round = 0

        self.client_uncertainties = [0.0] * num_clients
        self.client_data_sizes = [0] * num_clients
        self.uncertainty_history = {i: [] for i in range(num_clients)}

    def distribute_models(self) -> Tuple[List[nn.Module], List[nn.Module], List[nn.Module]]:
        """Distribute global models to all clients."""
        import gc

        client_models = []
        client_fins_i2t = []
        client_fins_t2i = []

        for i in range(self.num_clients):
            if i > 0 and i % 3 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            client_models.append(copy.deepcopy(self.global_model))
            client_fins_i2t.append(copy.deepcopy(self.global_fin_i2t))
            client_fins_t2i.append(copy.deepcopy(self.global_fin_t2i))

        return client_models, client_fins_i2t, client_fins_t2i

    def aggregate_round(
        self,
        client_models: List[nn.Module],
        client_fins_i2t: List[nn.Module],
        client_fins_t2i: List[nn.Module],
        multimodal_indices: List[int]
    ):
        """Aggregate models after a federated round."""
        self.current_round += 1

        for i, unc in enumerate(self.client_uncertainties):
            self.uncertainty_history[i].append(unc)

        if self.use_uq_weighting:
            if self.current_round <= self.warmup_rounds:
                warmup_progress = self.current_round / self.warmup_rounds
                effective_alpha = self.uq_alpha * warmup_progress
                print(f"  [UQ] Warmup round {self.current_round}/{self.warmup_rounds}, "
                      f"effective_alpha={effective_alpha:.3f}")
            else:
                effective_alpha = self.uq_alpha
                print(f"  [UQ] Using full uncertainty weighting (alpha={effective_alpha:.2f})")

            self.global_model = uq_weighted_fedavg_aggregate(
                self.global_model,
                client_models,
                self.client_uncertainties,
                self.client_data_sizes,
                multimodal_indices=multimodal_indices,
                alpha=effective_alpha,
                temperature=self.uq_temperature,
                multimodal_boost=self.multimodal_boost
            )
        else:
            print("  [Standard] Using uniform/data-size weighting")
            total_data = sum(self.client_data_sizes)
            data_weights = [n / total_data for n in self.client_data_sizes]
            self.global_model = fedavg_aggregate(
                self.global_model, client_models, client_weights=data_weights
            )

        self.global_fin_i2t, self.global_fin_t2i = aggregate_fins(
            self.global_fin_i2t,
            self.global_fin_t2i,
            client_fins_i2t,
            client_fins_t2i,
            multimodal_indices,
            client_uncertainties=self.client_uncertainties,
            use_uncertainty_weighting=False
        )

    def update_client_stats(self, client_id: int, uncertainty: float, data_size: int):
        """Update client statistics for UQ-weighted aggregation."""
        self.client_uncertainties[client_id] = uncertainty
        self.client_data_sizes[client_id] = data_size

    def get_aggregation_summary(self) -> Dict:
        """Get summary of current aggregation state."""
        return {
            'round': self.current_round,
            'use_uq': self.use_uq_weighting,
            'alpha': self.uq_alpha,
            'temperature': self.uq_temperature,
            'warmup_progress': min(self.current_round / self.warmup_rounds, 1.0),
            'client_uncertainties': self.client_uncertainties.copy(),
            'client_data_sizes': self.client_data_sizes.copy()
        }


if __name__ == "__main__":
    print("Testing Federated Aggregation...")
    from multimodal_model import get_model
    from probabilistic_models import ProbabilisticFIN

    print("Creating models...")
    global_model = get_model(pretrained=False)
    global_fin_i2t = ProbabilisticFIN()
    global_fin_t2i = ProbabilisticFIN()

    trainer = FederatedTrainer(
        global_model=global_model,
        global_fin_i2t=global_fin_i2t,
        global_fin_t2i=global_fin_t2i,
        num_clients=10,
        use_uq_weighting=True,
        uq_alpha=0.3,
        uq_temperature=0.5,
        warmup_rounds=5
    )

    print("\nDistributing models to clients...")
    client_models, client_fins_i2t, client_fins_t2i = trainer.distribute_models()
    print(f"Distributed {len(client_models)} models to clients")

    print("\nSimulating client training...")
    for i in range(10):
        uncertainty = np.random.uniform(0.05, 0.15) if i < 8 else np.random.uniform(0.01, 0.05)
        trainer.update_client_stats(client_id=i, uncertainty=uncertainty, data_size=1000 + i * 100)
        print(f"  Client {i}: uncertainty={uncertainty:.4f}, data_size={trainer.client_data_sizes[i]}")

    multimodal_indices = [8, 9]
    for r in range(1, 7):
        print(f"\n--- Testing Round {r} ---")
        client_models, client_fins_i2t, client_fins_t2i = trainer.distribute_models()
        trainer.aggregate_round(client_models, client_fins_i2t, client_fins_t2i, multimodal_indices)

    summary = trainer.get_aggregation_summary()
    print(f"\n--- Final Summary ---")
    print(f"Rounds completed: {summary['round']}")
    print(f"Mean uncertainty: {np.mean(summary['client_uncertainties']):.4f}")
    print("\n✓ Aggregation test completed!")
