"""
Embedding model and training loop for the self-supervised training.

The model (``DINOv3Encoder``) is the same class as in ``inference/utils.py``;
a checkpoint written here loads there. Keep the two copies identical.

The training signal is a semi-hard triplet loss on pairs of crops of the same
location: the anchor is a crop as it is, the positive the same crop rotated in
3D, and the negatives are the positives of the other samples in the batch.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Local clone of https://github.com/facebookresearch/dinov3, by default at the
# top level of this repository. Override with the DINOV3_REPO environment
# variable or the ``repo_dir`` argument of the encoder.
DEFAULT_DINOV3_REPO = os.environ.get(
    "DINOV3_REPO", str(Path(__file__).resolve().parent.parent / "dinov3"))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class DINOv3Encoder(nn.Module):
    """DINOv3 backbone applied slice by slice to a small 3D crop.

    The RGB patch embedding is replaced by a single-channel one (initialised
    with the channel mean of the pretrained weights), each z-slice is encoded
    independently, projected to ``embedding_dim`` and max-pooled over z. The
    output is L2-normalised, so a dot product of two embeddings is their cosine
    similarity.

    Parameters
    ----------
    embedding_dim : int
        Size of the output embedding.
    model : str
        Name of the DINOv3 hub entry point, e.g. ``"dinov3_vitb16"``.
    freeze_backbone : bool
        Freeze all backbone weights and train the projection head only.
    repo_dir : str or Path, optional
        Local clone of the DINOv3 repository. Defaults to
        ``DEFAULT_DINOV3_REPO``.
    """

    def __init__(self, embedding_dim=128, model="dinov3_vitb16",
                 freeze_backbone=False, repo_dir=None):
        super().__init__()

        self.backbone = torch.hub.load(
            str(DEFAULT_DINOV3_REPO if repo_dir is None else repo_dir),
            model,
            source="local",
            pretrained=True,
        )

        # swap the 3-channel patch embedding for a 1-channel one
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.backbone.patch_embed.proj = new_proj

        feat_dim = self.backbone.embed_dim  # 768 for ViT-B

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x):
        """x: [B, C, T, H, W] with T the z-slices. Returns [B, embedding_dim]."""
        B, C, T, H, W = x.shape

        # encode every slice on its own
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.backbone(x)            # [B*T, feat_dim]
        feats = self.fc(feats)              # [B*T, D]

        # max-pool over the slices
        feats = feats.view(B, T, -1)
        feats = feats.max(dim=1).values

        return F.normalize(feats, dim=-1)   # [B, D]


# ----------------------------------------------------------------------------
# loss and training
# ----------------------------------------------------------------------------

def batch_semi_hard_triplet_loss(emb_a, emb_p, emb_n, margin=0.2):
    """Triplet loss with semi-hard negative mining inside the batch.

    Sample ``i`` is pulled towards its positive ``emb_p[i]`` and pushed away
    from the negatives ``emb_n[j]`` of all other samples ``j != i``. Of those,
    the most similar negative that is still less similar than the positive,
    but by less than ``margin``, is used (semi-hard); if there is none, the
    most similar negative overall (hardest).

    Parameters
    ----------
    emb_a, emb_p, emb_n : Tensor [B, D] or [B, T, D]
        Anchor, positive and negative embeddings. With a slice dimension T,
        similarities are averaged over the slices (pairs of slices for the
        negatives).
    margin : float
        Required gap between positive and negative cosine similarity.

    Returns
    -------
    Tensor
        Mean loss over the batch.
    """
    # handle both [B, D] and [B, T, D]
    if emb_a.dim() == 2:
        emb_a = emb_a.unsqueeze(1)
        emb_p = emb_p.unsqueeze(1)
        emb_n = emb_n.unsqueeze(1)

    B, T, D = emb_a.shape
    device = emb_a.device

    emb_a = F.normalize(emb_a, dim=2)
    emb_p = F.normalize(emb_p, dim=2)
    emb_n = F.normalize(emb_n, dim=2)

    # positive similarity, per slice, then averaged: [B]
    sim_ap = torch.sum(emb_a * emb_p, dim=2)
    sim_ap = sim_ap.mean(dim=1)

    # similarity of every anchor to every negative, averaged over slice pairs
    emb_a_flat = emb_a.view(B * T, D)
    emb_n_flat = emb_n.view(B * T, D)
    sim_an_full = emb_a_flat @ emb_n_flat.T          # [B*T, B*T]
    sim_an_full = sim_an_full.view(B, T, B, T)
    sim_an = sim_an_full.mean(dim=(1, 3))            # [B, B]

    # a sample's own positive is not a negative
    eye = torch.eye(B, device=device, dtype=torch.bool)
    sim_an = sim_an.masked_fill(eye, -1e9)

    # semi-hard negatives: pos_sim - margin < neg_sim < pos_sim
    pos_sim_expanded = sim_ap.unsqueeze(1)           # [B, 1]
    semi_hard_mask = (
        (sim_an < pos_sim_expanded) &
        (sim_an > pos_sim_expanded - margin)
    )

    # most similar semi-hard negative per row ...
    semi_hard_neg = torch.where(
        semi_hard_mask,
        sim_an,
        torch.tensor(-1e9, device=device)
    ).max(dim=1).values

    # ... or the hardest negative if the row has no semi-hard one
    hardest_neg = sim_an.max(dim=1).values
    use_semi = semi_hard_mask.any(dim=1)
    chosen_neg = torch.where(use_semi, semi_hard_neg, hardest_neg)

    loss = F.relu(margin + chosen_neg - sim_ap)
    return loss.mean()


def train(model, loader, device, epochs=10, lr=1e-4, margin=0.2, patience=7, min_delta=1e-3):
    """Train ``model`` on pairs (crop, rotated crop) with Adam.

    The negatives are the detached positives of the other samples in the
    batch. Training stops early once the epoch loss has not improved by more
    than ``min_delta`` for ``patience`` epochs.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
        Yields batches ``(crop, rotated_crop)`` of shape [B, 1, Z, Y, X].
    device : str or torch.device
    epochs : int
        Maximum number of epochs.
    lr : float
        Learning rate.
    margin : float
        Margin of the triplet loss.
    patience, min_delta
        Early stopping.

    Returns
    -------
    losses : ndarray (epochs,)
        Mean loss per epoch; zero for the epochs skipped by early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = np.zeros(epochs)

    model.to(device)
    model.train()

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_stat, batch_rot in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_stat = batch_stat.to(device, non_blocking=True).float()
            batch_rot = batch_rot.to(device, non_blocking=True).float()

            emb_a = model(batch_stat)
            emb_p = model(batch_rot)
            emb_n = emb_p.detach()

            loss = batch_semi_hard_triplet_loss(emb_a, emb_p, emb_n, margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.cuda.empty_cache()
        epoch_loss = total_loss / len(loader)
        losses[epoch] = epoch_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[epoch]:.4f}")

        # early stopping
        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return losses
