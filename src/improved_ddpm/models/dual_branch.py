"""
Module 4 - DualBranchFASNetwork
=================================
Mang phan loai Face Anti-Spoofing nhanh kep (Dual-Branch):
    - Nhanh 1 (M_rgb): doc anh RGB thuong -> backbone (ResNet) -> MLP -> F_rgb
    - Nhanh 2 (M_amp): doc THANG ma tran Amplitude (tu FFT) -> backbone rieng -> MLP -> F_amp

Multi-task loss:
    1) Classification Loss: CrossEntropy tren feature ket hop [F_rgb, F_amp]
    2) Frequency Contrastive Loss: keo F_amp cung class (Live-Live / Attack-Attack)
       lai gan, day xa F_amp khac class (Live-Attack)  (kieu SupCon)
    3) Center Loss (alpha=0.2): gom cum feature ket hop theo tam moi class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ----------------------------------------------------------------------------
# Backbone con cho moi nhanh: ResNet18 (bo lop FC cuoi) + MLP projection head
# ----------------------------------------------------------------------------
class BranchEncoder(nn.Module):
    """
    Mot nhanh bat ky (RGB hoac Amplitude) deu dung cung kien truc:
    ResNet18 backbone (co the thay bang CLIP Image Encoder) -> MLP head -> feature vector.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 256, pretrained: bool = False):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Amplitude co the khong phai 3 channel (vd grayscale) -> sua conv1 dau vao
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        backbone.fc = nn.Identity()  # bo lop classification goc, lay feature 512-d
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)          # (B, 512)
        feat = self.mlp_head(feat)       # (B, feat_dim)
        feat = F.normalize(feat, dim=-1)  # chuan hoa L2 -> on dinh cho contrastive loss
        return feat


# ----------------------------------------------------------------------------
# Center Loss (Wen et al., 2016) - gom cum feature theo tam class
# ----------------------------------------------------------------------------
class CenterLoss(nn.Module):
    """
    Duy tri 1 "tam" (center) hoc duoc cho moi class, ep feature cua sample
    tien gan ve tam class tuong ung. alpha=0.2 la trong so ket hop vao loss tong.
    """

    def __init__(self, num_classes: int = 2, feat_dim: int = 512):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        features: (B, feat_dim) - feature ket hop cua sample (vd torch.cat([F_rgb,F_amp]))
        labels:   (B,) - nhan class (0=Live, 1=Attack)
        """
        centers_batch = self.centers[labels]              # (B, feat_dim)
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss


# ----------------------------------------------------------------------------
# Frequency Contrastive Loss (kieu Supervised Contrastive - SupCon) tren F_amp
# ----------------------------------------------------------------------------
def frequency_contrastive_loss(f_amp: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    """
    Ep F_amp cung nhan (Live-Live hoac Attack-Attack) gan nhau, khac nhan day xa nhau.
    f_amp: (B, D) - DA duoc L2-normalize (BranchEncoder da normalize san)
    labels: (B,) - 0=Live, 1=Attack
    """
    B = f_amp.shape[0]
    device = f_amp.device

    sim_matrix = torch.matmul(f_amp, f_amp.T) / temperature  # (B,B) cosine sim / T
    sim_matrix.fill_diagonal_(-1e9)  # bo tu-tuong-quan voi chinh no

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float().to(device)
    positive_mask.fill_diagonal_(0)

    # log-softmax tren tung dong (moi sample la 1 "anchor")
    log_prob = F.log_softmax(sim_matrix, dim=1)

    # Trung binh log-prob cua cac cap positive (tranh chia 0 neu 1 sample khong co positive trong batch)
    pos_counts = positive_mask.sum(dim=1).clamp(min=1.0)
    loss_per_anchor = -(positive_mask * log_prob).sum(dim=1) / pos_counts

    # Loai cac anchor khong co positive nao trong batch (vd batch chi co 1 sample/class)
    valid = (positive_mask.sum(dim=1) > 0).float()
    loss = (loss_per_anchor * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


# ----------------------------------------------------------------------------
# Mang Dual-Branch hoan chinh
# ----------------------------------------------------------------------------
class DualBranchFASNetwork(nn.Module):
    """Flexible RGB/FFT classifier used by the Stage-5 ablation variants.

    The historical name is retained for checkpoint compatibility. With the
    default ``use_rgb=True`` and ``use_amp=True`` the parameter names and tensor
    shapes are identical to the original dual-branch implementation.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        amp_in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = False,
        use_rgb: bool = True,
        use_amp: bool = True,
    ):
        super().__init__()
        if not use_rgb and not use_amp:
            raise ValueError("At least one of use_rgb/use_amp must be enabled")

        self.use_rgb = bool(use_rgb)
        self.use_amp = bool(use_amp)
        self.feat_dim = int(feat_dim)

        self.rgb_branch = (
            BranchEncoder(in_channels=3, feat_dim=feat_dim, pretrained=pretrained)
            if self.use_rgb
            else None
        )
        self.amp_branch = (
            BranchEncoder(in_channels=amp_in_channels, feat_dim=feat_dim, pretrained=pretrained)
            if self.use_amp
            else None
        )

        combined_dim = feat_dim * (int(self.use_rgb) + int(self.use_amp))
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        self.center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=combined_dim)

    @property
    def active_modalities(self) -> str:
        if self.use_rgb and self.use_amp:
            return "rgb+fft"
        return "rgb" if self.use_rgb else "fft"

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        """Differentiable scalar zero on the correct device and dtype."""
        return reference.sum() * 0.0

    def forward(self, rgb_image: torch.Tensor | None, amplitude: torch.Tensor | None):
        """Forward only the branches enabled by the selected ablation variant."""
        features = []
        f_rgb = None
        f_amp = None

        if self.use_rgb:
            if rgb_image is None:
                raise ValueError("RGB branch is enabled but rgb_image is None")
            f_rgb = self.rgb_branch(rgb_image)
            features.append(f_rgb)
        if self.use_amp:
            if amplitude is None:
                raise ValueError("FFT branch is enabled but amplitude is None")
            f_amp = self.amp_branch(amplitude)
            features.append(f_amp)

        f_combined = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        logits = self.classifier(f_combined)
        return {
            "logits": logits,
            "f_rgb": f_rgb,
            "f_amp": f_amp,
            "f_combined": f_combined,
        }

    @torch.no_grad()
    def predict(self, rgb_full: torch.Tensor | None, amplitude_full: torch.Tensor | None):
        """Full-image inference; patch and DDPM are training-only components."""
        self.eval()
        outputs = self.forward(rgb_full, amplitude_full)
        probs = F.softmax(outputs["logits"], dim=-1)
        outputs["probs"] = probs
        outputs["pred"] = probs.argmax(dim=-1)
        return outputs

    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        alpha_center: float = 0.2,
        lambda_contrastive: float = 1.0,
    ):
        """Single-stream objective used by E0, E1 and E2."""
        l_cls = F.cross_entropy(outputs["logits"], labels)

        if self.use_amp and float(lambda_contrastive) > 0.0:
            l_contrastive = frequency_contrastive_loss(outputs["f_amp"], labels)
        else:
            l_contrastive = self._zero_loss(outputs["logits"])

        if float(alpha_center) > 0.0:
            l_center = self.center_loss_fn(outputs["f_combined"], labels)
        else:
            l_center = self._zero_loss(outputs["logits"])

        l_total = l_cls + float(lambda_contrastive) * l_contrastive + float(alpha_center) * l_center
        return {
            "loss": l_total,
            "l_face": l_cls,
            "l_patch": self._zero_loss(outputs["logits"]),
            "l_cls": l_cls,
            "l_contrastive": l_contrastive,
            "l_center": l_center,
        }

    def compute_multi_stream_loss(
        self,
        outputs_full: dict,
        outputs_patch: dict,
        labels: torch.Tensor,
        alpha_center: float = 0.2,
        lambda_contrastive: float = 1.0,
    ):
        """Full + patch objective used by E3-E6."""
        l_face = F.cross_entropy(outputs_full["logits"], labels)
        l_patch = F.cross_entropy(outputs_patch["logits"], labels)
        l_cls = l_face + l_patch

        if self.use_amp and float(lambda_contrastive) > 0.0:
            f_amp_combined = torch.cat(
                [outputs_full["f_amp"], outputs_patch["f_amp"]], dim=0
            )
            labels_combined = torch.cat([labels, labels], dim=0)
            l_contrastive = frequency_contrastive_loss(f_amp_combined, labels_combined)
        else:
            labels_combined = torch.cat([labels, labels], dim=0)
            l_contrastive = self._zero_loss(outputs_full["logits"])

        if float(alpha_center) > 0.0:
            f_combined = torch.cat(
                [outputs_full["f_combined"], outputs_patch["f_combined"]], dim=0
            )
            l_center = self.center_loss_fn(f_combined, labels_combined)
        else:
            l_center = self._zero_loss(outputs_full["logits"])

        l_total = l_cls + float(lambda_contrastive) * l_contrastive + float(alpha_center) * l_center
        return {
            "loss": l_total,
            "l_face": l_face,
            "l_patch": l_patch,
            "l_cls": l_cls,
            "l_contrastive": l_contrastive,
            "l_center": l_center,
        }

