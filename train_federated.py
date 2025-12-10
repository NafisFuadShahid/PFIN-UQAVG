"""Main training script for Probabilistic Federated Feature Imputation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
import csv
import copy
from typing import Dict, List, Tuple

from data_loader_federated import create_federated_dataloaders, create_test_dataloader
from multimodal_model import get_model
from probabilistic_models import ProbabilisticFIN, DeterministicFIN, BetaNLLLoss, normalize_features
from federated_learning import FederatedTrainer


class LocalTrainer:
    """Handles local training at each client with uncertainty-aware fusion."""

    def __init__(self, client_id: int, is_multimodal: bool, model: nn.Module,
                 fin_i2t: nn.Module, fin_t2i: nn.Module,
                 use_probabilistic: bool = True, device: str = 'cuda'):
        self.client_id = client_id
        self.is_multimodal = is_multimodal
        self.model = model.to(device)
        self.fin_i2t = fin_i2t.to(device)
        self.fin_t2i = fin_t2i.to(device)
        self.use_probabilistic = use_probabilistic
        self.device = device

        pos_weight = torch.ones(14) * 8.0
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        if use_probabilistic:
            self.beta_nll_loss = BetaNLLLoss(beta=0.5)
        self.mse_loss = nn.MSELoss()

        self.mean_uncertainty = 0.0
        self.uncertainty_ema = 0.9

    def train_multimodal_client(self, dataloader: DataLoader, num_epochs: int = 3,
                                lr: float = 1e-4) -> float:
        """Train multimodal client: main model + FIN networks."""
        model_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        fin_optimizer = optim.Adam(
            list(self.fin_i2t.parameters()) + list(self.fin_t2i.parameters()), lr=lr
        )

        self.model.train()
        self.fin_i2t.train()
        self.fin_t2i.train()

        total_loss = 0.0
        num_batches = 0
        all_uncertainties = []

        for epoch in range(num_epochs):
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['labels'].to(self.device)

                # Train main model
                model_optimizer.zero_grad()
                img_feat, txt_feat, outputs = self.model(images, texts, use_uncertainty_weighting=False)
                logits = outputs['logits']
                cls_loss = self.bce_loss(logits, labels)
                cls_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optimizer.step()

                if num_batches % 10 == 0:
                    print(f"      Batch {num_batches}: cls_loss={cls_loss.item():.4f}")

                # Train FIN networks
                fin_optimizer.zero_grad()

                with torch.no_grad():
                    img_feat_target = self.model.encode_image(images)
                    txt_feat_target = self.model.encode_text(texts)

                if self.use_probabilistic:
                    alpha_t, beta_t, _ = self.fin_i2t(img_feat_target, target_features=txt_feat_target, return_normalized=True)
                    txt_feat_norm = normalize_features(txt_feat_target)
                    mean_t = alpha_t / (alpha_t + beta_t)
                    var_t = alpha_t * beta_t / ((alpha_t + beta_t)**2 * (alpha_t + beta_t + 1))
                    loss_i2t = self.beta_nll_loss(mean_t, var_t, txt_feat_norm)
                    unc_t = self.fin_i2t.get_uncertainty(alpha_t, beta_t).mean().item()

                    alpha_i, beta_i, _ = self.fin_t2i(txt_feat_target, target_features=img_feat_target, return_normalized=True)
                    img_feat_norm = normalize_features(img_feat_target)
                    mean_i = alpha_i / (alpha_i + beta_i)
                    var_i = alpha_i * beta_i / ((alpha_i + beta_i)**2 * (alpha_i + beta_i + 1))
                    loss_t2i = self.beta_nll_loss(mean_i, var_i, img_feat_norm)
                    unc_i = self.fin_t2i.get_uncertainty(alpha_i, beta_i).mean().item()
                else:
                    pred_t = self.fin_i2t(img_feat_target)
                    loss_i2t = self.mse_loss(pred_t, txt_feat_target)
                    pred_i = self.fin_t2i(txt_feat_target)
                    loss_t2i = self.mse_loss(pred_i, img_feat_target)
                    unc_t = unc_i = 0.0

                fin_loss = loss_i2t + loss_t2i
                fin_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fin_i2t.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.fin_t2i.parameters(), max_norm=1.0)
                fin_optimizer.step()

                total_loss += cls_loss.item()
                num_batches += 1

                if self.use_probabilistic:
                    all_uncertainties.append((unc_t + unc_i) / 2)

        self.mean_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0
        return self.mean_uncertainty

    def train_unimodal_client(self, dataloader: DataLoader, missing_modality: str = 'text',
                              num_epochs: int = 3, lr: float = 1e-4) -> float:
        """Train unimodal client with uncertainty-aware fusion."""
        model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        self.fin_i2t.eval()
        self.fin_t2i.eval()

        total_loss = 0.0
        num_batches = 0
        all_uncertainties = []

        for epoch in range(num_epochs):
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['labels'].to(self.device)

                model_optimizer.zero_grad()

                if missing_modality == 'text':
                    with torch.no_grad():
                        img_feat = self.model.encode_image(images)
                        if self.use_probabilistic:
                            txt_feat, txt_uncertainty, mean_unc = self.fin_i2t.impute_with_uncertainty(img_feat)
                            all_uncertainties.append(mean_unc.mean().item())
                        else:
                            txt_feat = self.fin_i2t(img_feat)
                            txt_uncertainty = None

                    img_feat = self.model.encode_image(images)
                    image_uncertainty = None
                    text_uncertainty = txt_uncertainty
                else:
                    with torch.no_grad():
                        txt_feat = self.model.encode_text(texts)
                        if self.use_probabilistic:
                            img_feat, img_uncertainty, mean_unc = self.fin_t2i.impute_with_uncertainty(txt_feat)
                            all_uncertainties.append(mean_unc.mean().item())
                        else:
                            img_feat = self.fin_t2i(txt_feat)
                            img_uncertainty = None

                    txt_feat = self.model.encode_text(texts)
                    text_uncertainty = None
                    image_uncertainty = img_uncertainty

                outputs = self.model.fusion_classifier(
                    img_feat, txt_feat,
                    image_uncertainty=image_uncertainty,
                    text_uncertainty=text_uncertainty,
                    use_uncertainty_weighting=self.use_probabilistic
                )
                logits = outputs['logits']

                loss = self.bce_loss(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optimizer.step()

                if num_batches % 10 == 0:
                    print(f"      Batch {num_batches}: cls_loss={loss.item():.4f}")

                total_loss += loss.item()
                num_batches += 1

        self.mean_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0
        return self.mean_uncertainty


def evaluate_model(model: nn.Module, fin_i2t: nn.Module, test_loader: DataLoader,
                   device: str = 'cuda', use_probabilistic: bool = True,
                   force_imputation: bool = False) -> Dict[str, float]:
    """Evaluate model on test set."""
    model = model.to(device)
    fin_i2t = fin_i2t.to(device)
    model.eval()
    fin_i2t.eval()

    all_preds = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['labels'].to(device)
            has_text = batch['has_text']

            img_feat = model.encode_image(images)

            if force_imputation or not any(has_text):
                if use_probabilistic:
                    txt_feat, text_uncertainty, _ = fin_i2t.impute_with_uncertainty(img_feat)
                    all_uncertainties.append(text_uncertainty.mean().item())
                else:
                    txt_feat = fin_i2t.impute(img_feat)
                    text_uncertainty = None
            else:
                txt_feat = model.encode_text(texts)
                text_uncertainty = None

            outputs = model.fusion_classifier(
                img_feat, txt_feat,
                image_uncertainty=None,
                text_uncertainty=text_uncertainty,
                use_uncertainty_weighting=use_probabilistic
            )
            logits = outputs['logits']
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("\n=== Evaluation Diagnostics ===")
    print(f"Prediction Stats: Min={all_preds.min():.4f}, Max={all_preds.max():.4f}, Mean={all_preds.mean():.4f}")
    print(f"Label Stats: Total samples={all_labels.shape[0]}")

    preds_binary = (all_preds > 0.5).astype(int)
    print(f"\nPer-class Prediction Distribution:")
    for i in range(min(14, all_labels.shape[1])):
        pred_pos_rate = preds_binary[:, i].mean()
        label_pos_rate = all_labels[:, i].mean()
        print(f"  Class {i}: Pred={pred_pos_rate:.3f}, Label={label_pos_rate:.3f}")

    try:
        auc_scores = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)
        macro_auc = np.mean(auc_scores) if auc_scores else 0.0
    except Exception as e:
        print(f"AUC computation error: {e}")
        macro_auc = 0.0

    try:
        macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    except Exception as e:
        print(f"F1 computation error: {e}")
        macro_f1 = 0.0

    mean_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0

    return {'auc': macro_auc, 'f1': macro_f1, 'mean_uncertainty': mean_uncertainty}


def save_checkpoint(round_num: int, model: nn.Module, fin_i2t: nn.Module,
                    fin_t2i: nn.Module, metrics: Dict, save_dir: str = 'checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'round': round_num,
        'model_state': model.state_dict(),
        'fin_i2t_state': fin_i2t.state_dict(),
        'fin_t2i_state': fin_t2i.state_dict(),
        'metrics': metrics
    }
    path = os.path.join(save_dir, f'checkpoint_round_{round_num}.pth')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def pretrain_fin(global_model: nn.Module, global_fin_i2t: nn.Module, global_fin_t2i: nn.Module,
                 multimodal_loaders: List[DataLoader], num_epochs: int = 5, lr: float = 1e-4,
                 device: str = 'cuda', use_probabilistic: bool = True):
    """Pre-train FIN networks on multimodal data."""
    print("\n=== Pre-training FIN on Multimodal Data ===")

    global_model = global_model.to(device)
    global_fin_i2t = global_fin_i2t.to(device)
    global_fin_t2i = global_fin_t2i.to(device)

    global_model.eval()
    global_fin_i2t.train()
    global_fin_t2i.train()

    fin_optimizer = optim.Adam(
        list(global_fin_i2t.parameters()) + list(global_fin_t2i.parameters()), lr=lr
    )

    if use_probabilistic:
        beta_nll_loss = BetaNLLLoss(beta=0.5)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for loader in multimodal_loaders:
            for batch in loader:
                images = batch['image'].to(device)
                texts = batch['text']

                fin_optimizer.zero_grad()

                with torch.no_grad():
                    img_feat = global_model.encode_image(images)
                    txt_feat = global_model.encode_text(texts)

                if use_probabilistic:
                    alpha_t, beta_t, _ = global_fin_i2t(img_feat, target_features=txt_feat, return_normalized=True)
                    txt_feat_norm = normalize_features(txt_feat)
                    mean_t = alpha_t / (alpha_t + beta_t)
                    var_t = alpha_t * beta_t / ((alpha_t + beta_t)**2 * (alpha_t + beta_t + 1))
                    loss_i2t = beta_nll_loss(mean_t, var_t, txt_feat_norm)

                    alpha_i, beta_i, _ = global_fin_t2i(txt_feat, target_features=img_feat, return_normalized=True)
                    img_feat_norm = normalize_features(img_feat)
                    mean_i = alpha_i / (alpha_i + beta_i)
                    var_i = alpha_i * beta_i / ((alpha_i + beta_i)**2 * (alpha_i + beta_i + 1))
                    loss_t2i = beta_nll_loss(mean_i, var_i, img_feat_norm)
                else:
                    pred_t = global_fin_i2t(img_feat)
                    loss_i2t = mse_loss(pred_t, txt_feat)
                    pred_i = global_fin_t2i(txt_feat)
                    loss_t2i = mse_loss(pred_i, img_feat)

                fin_loss = loss_i2t + loss_t2i
                fin_loss.backward()

                torch.nn.utils.clip_grad_norm_(global_fin_i2t.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(global_fin_t2i.parameters(), max_norm=1.0)

                fin_optimizer.step()
                total_loss += fin_loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  FIN Pre-training Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    global_fin_i2t.eval()
    with torch.no_grad():
        test_img = torch.randn(4, 256).to(device)
        if use_probabilistic:
            alpha, beta, imputed = global_fin_i2t(test_img, return_normalized=False)
            unc = global_fin_i2t.get_uncertainty(alpha, beta).mean().item()
            print(f"  Post-pretraining uncertainty: {unc:.6f}")

    print("  FIN Pre-training complete!\n")
    return global_fin_i2t, global_fin_t2i


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n=== Creating Federated Dataloaders ===")
    client_loaders, is_multimodal_list = create_federated_dataloaders(
        num_multimodal_clients=2, num_unimodal_clients=8, batch_size=args.batch_size
    )
    test_loader = create_test_dataloader(batch_size=args.batch_size)

    print("\n=== Creating Models ===")
    global_model = get_model(use_uncertainty=args.use_probabilistic, pretrained=True, fusion_type='attention')

    print("\n=== Parameter Check ===")
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

    if args.use_probabilistic:
        global_fin_i2t = ProbabilisticFIN()
        global_fin_t2i = ProbabilisticFIN()
        print("Using Probabilistic FIN with Beta-NLL")
    else:
        global_fin_i2t = DeterministicFIN()
        global_fin_t2i = DeterministicFIN()
        print("Using Deterministic FIN with MSE")

    if args.pretrain_fin_epochs > 0:
        multimodal_loaders = [client_loaders[i] for i, is_mm in enumerate(is_multimodal_list) if is_mm]
        global_fin_i2t, global_fin_t2i = pretrain_fin(
            global_model=global_model, global_fin_i2t=global_fin_i2t, global_fin_t2i=global_fin_t2i,
            multimodal_loaders=multimodal_loaders, num_epochs=args.pretrain_fin_epochs,
            lr=args.lr, device=device, use_probabilistic=args.use_probabilistic
        )

    fed_trainer = FederatedTrainer(
        global_model=global_model, global_fin_i2t=global_fin_i2t, global_fin_t2i=global_fin_t2i,
        num_clients=10, use_uq_weighting=args.use_uq_weighting, uq_alpha=args.uq_alpha,
        uq_temperature=args.uq_temperature, warmup_rounds=args.warmup_rounds,
        multimodal_boost=args.multimodal_boost
    )

    print(f"\n=== Starting Federated Training for {args.num_rounds} rounds ===")
    print(f"  UQ Weighting: {args.use_uq_weighting}, Alpha: {args.uq_alpha}, Temp: {args.uq_temperature}")

    best_auc = 0.0
    results = []

    for round_num in range(args.num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{args.num_rounds}")
        print(f"{'='*60}")

        # Stage 1: Train multimodal clients
        print("\n--- Stage 1: Training Multimodal Clients ---")
        client_models, client_fins_i2t, client_fins_t2i = fed_trainer.distribute_models()
        multimodal_indices = [i for i, is_mm in enumerate(is_multimodal_list) if is_mm]

        for client_id in multimodal_indices:
            print(f"\nClient {client_id} (Multimodal):")
            trainer = LocalTrainer(
                client_id=client_id, is_multimodal=True, model=client_models[client_id],
                fin_i2t=client_fins_i2t[client_id], fin_t2i=client_fins_t2i[client_id],
                use_probabilistic=args.use_probabilistic, device=device
            )
            uncertainty = trainer.train_multimodal_client(
                client_loaders[client_id], num_epochs=args.local_epochs, lr=args.lr
            )
            data_size = len(client_loaders[client_id].dataset)
            fed_trainer.update_client_stats(client_id, uncertainty, data_size)
            print(f"  Uncertainty: {uncertainty:.6f}, Data size: {data_size}")

        # Aggregate FINs from multimodal clients
        print("\n--- Aggregating FINs from Multimodal Clients ---")
        from federated_learning import aggregate_fins
        fed_trainer.global_fin_i2t, fed_trainer.global_fin_t2i = aggregate_fins(
            fed_trainer.global_fin_i2t, fed_trainer.global_fin_t2i,
            client_fins_i2t, client_fins_t2i, multimodal_indices
        )

        # Stage 2: Train unimodal clients
        print("\n--- Stage 2: Training Unimodal Clients ---")
        unimodal_indices = [i for i, is_mm in enumerate(is_multimodal_list) if not is_mm]

        for client_id in unimodal_indices:
            print(f"\nClient {client_id} (Unimodal):")
            trainer = LocalTrainer(
                client_id=client_id, is_multimodal=False, model=client_models[client_id],
                fin_i2t=copy.deepcopy(fed_trainer.global_fin_i2t),
                fin_t2i=copy.deepcopy(fed_trainer.global_fin_t2i),
                use_probabilistic=args.use_probabilistic, device=device
            )
            uncertainty = trainer.train_unimodal_client(
                client_loaders[client_id], missing_modality='text',
                num_epochs=args.local_epochs, lr=args.lr
            )
            data_size = len(client_loaders[client_id].dataset)
            fed_trainer.update_client_stats(client_id, uncertainty, data_size)
            print(f"  Uncertainty: {uncertainty:.6f}, Data size: {data_size}")

        # Aggregate models
        print("\n--- Aggregating models ---")
        fed_trainer.aggregate_round(client_models, client_fins_i2t, client_fins_t2i, multimodal_indices)

        # Save client uncertainties
        os.makedirs(args.save_dir, exist_ok=True)
        agg_summary = fed_trainer.get_aggregation_summary()
        client_uncertainties = agg_summary['client_uncertainties']
        mean_unc = np.mean(client_uncertainties) if client_uncertainties else 0.0

        client_unc_csv_path = os.path.join(args.save_dir, 'client_uncertainties.csv')
        write_header = not os.path.exists(client_unc_csv_path) or round_num == 0
        with open(client_unc_csv_path, 'a' if round_num > 0 else 'w', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['round'] + [f'client_{i}_unc' for i in range(len(client_uncertainties))] + ['mean_unc']
                writer.writerow(header)
            row = [round_num + 1] + [f"{unc:.6f}" for unc in client_uncertainties] + [f"{mean_unc:.6f}"]
            writer.writerow(row)

        # Evaluate
        if (round_num + 1) % args.eval_every == 0:
            print("\n--- Evaluating on test set ---")
            metrics = evaluate_model(
                fed_trainer.global_model, fed_trainer.global_fin_i2t, test_loader,
                device=device, use_probabilistic=args.use_probabilistic, force_imputation=False
            )

            print(f"Test AUC: {metrics['auc']:.4f}, Test F1: {metrics['f1']:.4f}")

            results.append({
                'round': round_num + 1, 'auc': metrics['auc'], 'f1': metrics['f1'],
                'mean_uncertainty': metrics.get('mean_uncertainty', 0.0),
                'client_mean_uncertainty': mean_unc
            })

            csv_path = os.path.join(args.save_dir, 'training_results.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['round', 'test_auc', 'test_f1', 'eval_uncertainty', 'client_mean_uncertainty'])
                for r in results:
                    writer.writerow([r['round'], f"{r['auc']:.4f}", f"{r['f1']:.4f}",
                                   f"{r['mean_uncertainty']:.6f}", f"{r.get('client_mean_uncertainty', 0.0):.6f}"])
            print(f"Results saved to {csv_path}")

            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                save_checkpoint(
                    round_num + 1, fed_trainer.global_model, fed_trainer.global_fin_i2t,
                    fed_trainer.global_fin_t2i, metrics, save_dir=args.save_dir
                )

    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training completed! Best AUC: {best_auc:.4f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_rounds', type=int, default=30)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--use_probabilistic', action='store_true', default=False)
    parser.add_argument('--use_uq_weighting', action='store_true', default=False)

    parser.add_argument('--uq_alpha', type=float, default=0.3)
    parser.add_argument('--uq_temperature', type=float, default=0.5)
    parser.add_argument('--warmup_rounds', type=int, default=5)
    parser.add_argument('--multimodal_boost', type=float, default=3.0)
    parser.add_argument('--pretrain_fin_epochs', type=int, default=10)

    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints_prob')

    args = parser.parse_args()
    main(args)
