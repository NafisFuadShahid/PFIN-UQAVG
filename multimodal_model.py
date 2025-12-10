"""Multimodal Model with Uncertainty Quantification and Attention-Based Fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer
from typing import Dict, Tuple, Optional, List
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
warnings.filterwarnings('ignore')

ATTENTION_WEIGHTS_STORAGE = {'weights': [], 'uncertainties': [], 'sample_ids': []}


class ImageEncoder(nn.Module):
    """ResNet-50 based image encoder with L2-normalized output."""

    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        features = F.normalize(features, p=2, dim=-1)
        return features


class TextEncoder(nn.Module):
    """BERT-base text encoder with L2-normalized output."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(768, output_dim)
        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        features = self.projection(cls_output)
        features = F.normalize(features, p=2, dim=-1)
        return features


class UncertaintyAwareFusion(nn.Module):
    """Cross-modal attention fusion using nn.MultiheadAttention with uncertainty integration."""

    def __init__(self, feature_dim: int = 256, num_heads: int = 4, dropout: float = 0.1,
                 fusion_type: str = 'attention'):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.fusion_type = fusion_type

        # Cross-modal attention: Image attends to Text
        self.cross_attn_i2t = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-modal attention: Text attends to Image
        self.cross_attn_t2i = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Self-attention on concatenated features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms for attention outputs
        self.norm_i2t = nn.LayerNorm(feature_dim)
        self.norm_t2i = nn.LayerNorm(feature_dim)
        self.norm_self = nn.LayerNorm(feature_dim)

        # Uncertainty-aware gating
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.Sigmoid()
        )

        # Feed-forward for each modality after attention
        self.ffn_image = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _apply_uncertainty_mask(self, attn_weights: torch.Tensor, uncertainty: torch.Tensor,
                                 batch_size: int, device: torch.device) -> torch.Tensor:
        """Apply uncertainty-based soft masking to attention weights."""
        if uncertainty is None:
            return attn_weights

        if uncertainty.dim() > 1:
            uncertainty = uncertainty.mean(dim=-1)

        # Higher uncertainty -> lower attention (soft mask)
        confidence = torch.exp(-uncertainty * 5.0)  # Scale factor for sensitivity
        confidence = confidence.view(batch_size, 1, 1).expand_as(attn_weights)
        return attn_weights * confidence

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_uncertainty: Optional[torch.Tensor] = None,
        text_uncertainty: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = True,
        store_attention: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse image and text features using MultiheadAttention."""
        batch_size = image_features.size(0)
        device = image_features.device
        info = {}

        # Reshape for attention: [B, 1, D] (single token per modality)
        img_seq = image_features.unsqueeze(1)  # [B, 1, D]
        txt_seq = text_features.unsqueeze(1)   # [B, 1, D]

        # Cross-modal attention: Image queries Text
        img_attended, attn_i2t = self.cross_attn_i2t(
            query=img_seq, key=txt_seq, value=txt_seq, need_weights=True
        )
        img_attended = self.norm_i2t(img_seq + img_attended)

        # Cross-modal attention: Text queries Image
        txt_attended, attn_t2i = self.cross_attn_t2i(
            query=txt_seq, key=img_seq, value=img_seq, need_weights=True
        )
        txt_attended = self.norm_t2i(txt_seq + txt_attended)

        # Apply uncertainty weighting if enabled
        if use_uncertainty_weighting:
            if text_uncertainty is not None:
                # Reduce image's attention to uncertain text
                text_conf = torch.exp(-text_uncertainty.mean(dim=-1) if text_uncertainty.dim() > 1 else -text_uncertainty)
                text_gate = text_conf.view(batch_size, 1, 1)
                img_attended = img_attended * text_gate + img_seq * (1 - text_gate)

            if image_uncertainty is not None:
                # Reduce text's attention to uncertain image
                img_conf = torch.exp(-image_uncertainty.mean(dim=-1) if image_uncertainty.dim() > 1 else -image_uncertainty)
                img_gate = img_conf.view(batch_size, 1, 1)
                txt_attended = txt_attended * img_gate + txt_seq * (1 - img_gate)

        # FFN for each modality
        img_out = img_attended + self.ffn_image(img_attended)
        txt_out = txt_attended + self.ffn_text(txt_attended)

        # Concatenate and apply self-attention
        combined = torch.cat([img_out, txt_out], dim=1)  # [B, 2, D]
        combined_attended, attn_self = self.self_attn(
            query=combined, key=combined, value=combined, need_weights=True
        )
        combined_out = self.norm_self(combined + combined_attended)

        # Extract final representations
        final_img = combined_out[:, 0, :]  # [B, D]
        final_txt = combined_out[:, 1, :]  # [B, D]

        # Concatenate and project
        fused = torch.cat([final_img, final_txt], dim=-1)  # [B, 2D]
        fused = self.fusion_proj(fused)

        # Build attention map for visualization (average of cross-attention weights)
        attn_map = torch.zeros(batch_size, 2, 2, device=device)
        attn_map[:, 0, 1] = attn_i2t.squeeze(1).squeeze(1)  # Image -> Text attention
        attn_map[:, 1, 0] = attn_t2i.squeeze(1).squeeze(1)  # Text -> Image attention
        attn_map[:, 0, 0] = attn_self[:, 0, 0]  # Image self
        attn_map[:, 1, 1] = attn_self[:, 1, 1]  # Text self

        # Compute modality weights from self-attention
        modality_weights = attn_self.mean(dim=1)  # [B, 2]
        info['image_weight'] = modality_weights[:, 0]
        info['text_weight'] = modality_weights[:, 1]
        info['attention_map'] = attn_map
        info['cross_attn_i2t'] = attn_i2t
        info['cross_attn_t2i'] = attn_t2i

        if image_uncertainty is not None:
            info['image_uncertainty'] = image_uncertainty.mean(dim=-1) if image_uncertainty.dim() > 1 else image_uncertainty
        if text_uncertainty is not None:
            info['text_uncertainty'] = text_uncertainty.mean(dim=-1) if text_uncertainty.dim() > 1 else text_uncertainty

        if store_attention:
            global ATTENTION_WEIGHTS_STORAGE
            ATTENTION_WEIGHTS_STORAGE['weights'].append(attn_map.detach().cpu())
            if image_uncertainty is not None:
                img_unc = image_uncertainty.mean(dim=-1) if image_uncertainty.dim() > 1 else image_uncertainty
                txt_unc = text_uncertainty.mean(dim=-1) if text_uncertainty is not None and text_uncertainty.dim() > 1 else text_uncertainty
                ATTENTION_WEIGHTS_STORAGE['uncertainties'].append(
                    torch.stack([img_unc, txt_unc if txt_unc is not None else torch.zeros_like(img_unc)], dim=-1).detach().cpu()
                )

        return fused, info


class MultimodalFusionClassifier(nn.Module):
    """Multimodal fusion + classifier with MultiheadAttention-based uncertainty-aware fusion."""

    def __init__(self, feature_dim: int = 256, num_classes: int = 14,
                 use_uncertainty: bool = True, fusion_type: str = 'attention',
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        self.fusion = UncertaintyAwareFusion(
            feature_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, fusion_type=fusion_type
        )
        fused_dim = 2 * feature_dim

        if use_uncertainty:
            self.fc_logits = nn.Linear(fused_dim, num_classes)
            self.fc_alpha = nn.Linear(fused_dim, num_classes)
            self.fc_beta = nn.Linear(fused_dim, num_classes)
        else:
            self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_uncertainty: Optional[torch.Tensor] = None,
        text_uncertainty: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = True
    ) -> Dict[str, torch.Tensor]:
        fused, fusion_info = self.fusion(
            image_features, text_features, image_uncertainty, text_uncertainty, use_uncertainty_weighting
        )

        if self.use_uncertainty:
            logits = self.fc_logits(fused)
            alpha = F.softplus(self.fc_alpha(fused)) + 0.1
            beta = F.softplus(self.fc_beta(fused)) + 0.1
            return {'logits': logits, 'alpha': alpha, 'beta': beta, 'fused_features': fused, **fusion_info}
        else:
            logits = self.classifier(fused)
            return {'logits': logits, 'fused_features': fused, **fusion_info}


class MultimodalModel(nn.Module):
    """Complete multimodal model with MultiheadAttention-based uncertainty-aware fusion."""

    def __init__(self, feature_dim: int = 256, num_classes: int = 14,
                 use_uncertainty: bool = True, pretrained: bool = True,
                 fusion_type: str = 'attention', num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        self.image_encoder = ImageEncoder(output_dim=feature_dim, pretrained=pretrained)
        self.text_encoder = TextEncoder(output_dim=feature_dim)
        self.fusion_classifier = MultimodalFusionClassifier(
            feature_dim=feature_dim, num_classes=num_classes,
            use_uncertainty=use_uncertainty, fusion_type=fusion_type,
            num_heads=num_heads, dropout=dropout
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, texts: list) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoded['input_ids'].to(next(self.text_encoder.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.text_encoder.parameters()).device)
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        texts: list,
        image_uncertainty: Optional[torch.Tensor] = None,
        text_uncertainty: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        outputs = self.fusion_classifier(
            image_features, text_features, image_uncertainty, text_uncertainty, use_uncertainty_weighting
        )
        return image_features, text_features, outputs

    def forward_with_imputed(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_uncertainty: Optional[torch.Tensor] = None,
        text_uncertainty: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = True
    ) -> Dict:
        """Forward pass with pre-computed (possibly imputed) features."""
        return self.fusion_classifier(
            image_features, text_features, image_uncertainty, text_uncertainty, use_uncertainty_weighting
        )


def get_model(feature_dim: int = 256, num_classes: int = 14, use_uncertainty: bool = True,
              pretrained: bool = True, fusion_type: str = 'attention',
              num_heads: int = 4, dropout: float = 0.1) -> MultimodalModel:
    """Factory function to create model with MultiheadAttention fusion."""
    return MultimodalModel(
        feature_dim=feature_dim, num_classes=num_classes, use_uncertainty=use_uncertainty,
        pretrained=pretrained, fusion_type=fusion_type, num_heads=num_heads, dropout=dropout
    )


# Visualization utilities
def clear_attention_storage():
    global ATTENTION_WEIGHTS_STORAGE
    ATTENTION_WEIGHTS_STORAGE = {'weights': [], 'uncertainties': [], 'sample_ids': []}


def get_attention_weights() -> Dict:
    return ATTENTION_WEIGHTS_STORAGE


def plot_attention_heatmap(attention_map: torch.Tensor, title: str = "Cross-Modal Attention",
                           save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    labels = ['Image', 'Text']

    sns.heatmap(attention_map, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Attention Weight'})

    ax.set_xlabel('Key (Attended To)', fontsize=12)
    ax.set_ylabel('Query (Attending From)', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    return fig


def plot_attention_comparison(attention_multimodal: torch.Tensor, attention_unimodal: torch.Tensor,
                              uncertainty_unimodal: Optional[Tuple[float, float]] = None,
                              save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    if isinstance(attention_multimodal, torch.Tensor):
        attention_multimodal = attention_multimodal.detach().cpu().numpy()
    if isinstance(attention_unimodal, torch.Tensor):
        attention_unimodal = attention_unimodal.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = ['Image', 'Text']

    sns.heatmap(attention_multimodal, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0], vmin=0, vmax=1,
                cbar_kws={'label': 'Attention Weight'})
    axes[0].set_xlabel('Key (Attended To)', fontsize=11)
    axes[0].set_ylabel('Query (Attending From)', fontsize=11)
    axes[0].set_title('Multimodal Client\n(Both Modalities Observed)', fontsize=12)

    sns.heatmap(attention_unimodal, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1,
                cbar_kws={'label': 'Attention Weight'})
    axes[1].set_xlabel('Key (Attended To)', fontsize=11)
    axes[1].set_ylabel('Query (Attending From)', fontsize=11)

    if uncertainty_unimodal:
        title = f'Unimodal Client (Text Imputed)\nUncertainty: Img={uncertainty_unimodal[0]:.3f}, Txt={uncertainty_unimodal[1]:.3f}'
    else:
        title = 'Unimodal Client\n(Text Imputed with High Uncertainty)'
    axes[1].set_title(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention comparison to {save_path}")

    return fig


def plot_attention_by_uncertainty(attention_weights_list: List[torch.Tensor],
                                   uncertainties_list: List[Tuple[float, float]],
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    attn_data = []
    for attn, (img_unc, txt_unc) in zip(attention_weights_list, uncertainties_list):
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        img_attn_weight = attn[:, 0].mean()
        txt_attn_weight = attn[:, 1].mean()
        attn_data.append({
            'img_unc': img_unc, 'txt_unc': txt_unc,
            'img_attn': img_attn_weight, 'txt_attn': txt_attn_weight
        })

    txt_uncs = [d['txt_unc'] for d in attn_data]
    img_attns = [d['img_attn'] for d in attn_data]
    txt_attns = [d['txt_attn'] for d in attn_data]

    axes[0, 0].scatter(txt_uncs, txt_attns, c='orange', alpha=0.7, s=50)
    axes[0, 0].set_xlabel('Text Uncertainty', fontsize=11)
    axes[0, 0].set_ylabel('Text Attention Weight', fontsize=11)
    axes[0, 0].set_title('Text Attention vs Text Uncertainty', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(txt_uncs, img_attns, c='blue', alpha=0.7, s=50)
    axes[0, 1].set_xlabel('Text Uncertainty', fontsize=11)
    axes[0, 1].set_ylabel('Image Attention Weight', fontsize=11)
    axes[0, 1].set_title('Image Attention vs Text Uncertainty', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    attn_ratios = [img / max(txt, 1e-6) for img, txt in zip(img_attns, txt_attns)]
    axes[1, 0].scatter(txt_uncs, attn_ratios, c='green', alpha=0.7, s=50)
    axes[1, 0].set_xlabel('Text Uncertainty', fontsize=11)
    axes[1, 0].set_ylabel('Attention Ratio (Image/Text)', fontsize=11)
    axes[1, 0].set_title('Attention Ratio vs Text Uncertainty', fontsize=12)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal attention')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    n_samples = min(10, len(attn_data))
    sample_indices = np.linspace(0, len(attn_data)-1, n_samples, dtype=int)
    x = np.arange(n_samples)
    width = 0.6

    img_vals = [img_attns[i] for i in sample_indices]
    txt_vals = [txt_attns[i] for i in sample_indices]

    axes[1, 1].bar(x, img_vals, width, label='Image Attention', color='steelblue')
    axes[1, 1].bar(x, txt_vals, width, bottom=img_vals, label='Text Attention', color='coral')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Attention Weight', fontsize=11)
    axes[1, 1].set_title('Attention Distribution Across Samples', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(i) for i in sample_indices])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention analysis to {save_path}")

    return fig


def generate_attention_heatmap_for_paper(model: nn.Module, image_features: torch.Tensor,
                                          text_features: torch.Tensor, text_uncertainty: torch.Tensor,
                                          save_dir: str = "paper_results",
                                          filename: str = "attention_heatmap.png") -> str:
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'fusion_classifier'):
            outputs = model.fusion_classifier(
                image_features, text_features,
                image_uncertainty=None, text_uncertainty=text_uncertainty,
                use_uncertainty_weighting=True
            )
        else:
            outputs = model.fusion(
                image_features, text_features,
                image_uncertainty=None, text_uncertainty=text_uncertainty,
                use_uncertainty_weighting=True
            )
            if isinstance(outputs, tuple):
                _, outputs = outputs[0], outputs[1]

    attention_map = outputs.get('attention_map', None)
    if attention_map is None:
        print("Warning: No attention map found")
        return None

    avg_attention = attention_map.mean(dim=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Image\n(Observed)', 'Text\n(Imputed)']

    if isinstance(avg_attention, torch.Tensor):
        avg_attention = avg_attention.detach().cpu().numpy()

    sns.heatmap(avg_attention, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=ax, vmin=0, vmax=1,
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.8})

    ax.set_xlabel('Key (Attended To)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Query (Attending From)', fontsize=13, fontweight='bold')
    ax.set_title('Uncertainty-Aware Cross-Modal Attention\nfor Unimodal Client (Text Imputed)',
                fontsize=14, fontweight='bold')

    mean_txt_unc = text_uncertainty.mean().item() if isinstance(text_uncertainty, torch.Tensor) else text_uncertainty
    ax.text(0.5, -0.15, f'Text Uncertainty: {mean_txt_unc:.4f}',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')

    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved attention heatmap to {save_path}")
    return save_path


if __name__ == "__main__":
    print("Testing Multimodal Model...")

    model = get_model(use_uncertainty=True, pretrained=False, fusion_type='attention')

    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["Test report 1.", "Test report 2.", "Test report 3.", "Test report 4."]

    print("\n1. Standard forward pass:")
    img_feat, txt_feat, outputs = model(images, texts, use_uncertainty_weighting=False)
    print(f"  Image features: {img_feat.shape}")
    print(f"  Text features: {txt_feat.shape}")
    print(f"  Logits: {outputs['logits'].shape}")

    print("\n2. Forward with uncertainty:")
    text_uncertainty = torch.ones(batch_size) * 0.1
    image_uncertainty = torch.ones(batch_size) * 0.01

    _, _, outputs_uq = model(images, texts, image_uncertainty=image_uncertainty,
                             text_uncertainty=text_uncertainty, use_uncertainty_weighting=True)
    print(f"  Image weight: {outputs_uq['image_weight'].mean():.3f}")
    print(f"  Text weight: {outputs_uq['text_weight'].mean():.3f}")

    print("\n3. L2 normalization check:")
    img_norms = torch.norm(img_feat, p=2, dim=-1)
    txt_norms = torch.norm(txt_feat, p=2, dim=-1)
    print(f"  Image norms (should be ~1.0): {img_norms.mean():.4f}")
    print(f"  Text norms (should be ~1.0): {txt_norms.mean():.4f}")

    print("\n✓ Model test completed!")
