import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchmetrics import MeanMetric

from src.data.loading.components.interfaces import ItemData
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.modules.clustering.vector_quantization import VectorQuantization


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion producing a shared commonality embedding.

    z_t attends to z_i (captures image context for text), and z_i attends to z_t
    (captures text context for image). The two enriched representations are
    concatenated and projected to produce z_f.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.t2i_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.i2t_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.proj = nn.Linear(2 * dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, z_t: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: text latent   (B, d)
            z_i: image latent  (B, d)
        Returns:
            z_f: commonality embedding (B, d)
        """
        # MHA requires a sequence dimension
        q_t = z_t.unsqueeze(1)  # (B, 1, d)
        q_i = z_i.unsqueeze(1)

        ctx_t, _ = self.t2i_attn(q_t, q_i, q_i)  # text queries image context
        ctx_i, _ = self.i2t_attn(q_i, q_t, q_t)  # image queries text context

        h_t = self.norm_t(z_t + ctx_t.squeeze(1))
        h_i = self.norm_i(z_i + ctx_i.squeeze(1))

        return self.out_norm(self.proj(torch.cat([h_t, h_i], dim=-1)))


class MDRQVAE(LightningModule):
    """Multi-modal Dual Residual Quantized VAE.

    Produces three semantic IDs per item from paired text and image features:
      - ID_shared : commonality (shared codebook C_f on z_f)
      - ID_text   : text-specific (codebook C_t on MLP([r_f; z_t]))
      - ID_img    : image-specific (codebook C_i on MLP([r_f; z_i]))

    Forward path:
      z_t = E_t(x_t),  z_i = E_i(x_i)
      z_f = F(z_t, z_i)                           [fusion layer]
      e_f1 ~ C_f(z_f),    r_f = z_f - e_f1        [shared level]
      e_t2 ~ C_t(MLP([r_f; z_t]))                  [text-specific level]
      e_i3 ~ C_i(MLP([r_f; z_i]))                  [image-specific level]
      x_t_hat = D_t(e_f1 + e_t2)
      x_i_hat = D_i(e_f1 + e_i3)

    Loss:
      L_total = L_rec + alpha * L_aux + beta * L_vq
      L_rec   = MSE(x_t, x_t_hat) + MSE(x_i, x_i_hat)
      L_aux   = InfoNCE(z_f, z_t)  + InfoNCE(z_f, z_i)   [learnable temperature]
      L_vq    = BetaQuantizationLoss summed over {C_f, C_t, C_i}
                (commitment weight gamma lives in each codebook's loss_function.beta)

    Expected batch keys (ItemData.transformed_features):
      "text_embedding"  – raw text feature  (B, text_dim)
      "image_embedding" – raw image feature (B, img_dim)
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion_layer: nn.Module,
        text_fusion_mlp: nn.Module,
        image_fusion_mlp: nn.Module,
        shared_codebook: VectorQuantization,
        text_codebook: VectorQuantization,
        image_codebook: VectorQuantization,
        text_decoder: nn.Module,
        image_decoder: nn.Module,
        embedding_dim: int = 1152,
        alpha: float = 1.0,
        beta: float = 1.0,
        init_buffer_size: int = 1000,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        training_loop_function: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            text_encoder: E_t — maps raw text features to latent space.
            image_encoder: E_i — maps raw image features to latent space.
            fusion_layer: F — fuses (z_t, z_i) into shared z_f. Use
                CrossAttentionFusion or any nn.Module with signature (z_t, z_i) -> z_f.
            text_fusion_mlp: MLP that maps cat([r_f, z_t]) -> v_t for text codebook input.
                Input dim must equal 2 * latent_dim.
            image_fusion_mlp: MLP that maps cat([r_f, z_i]) -> v_i for image codebook input.
                Input dim must equal 2 * latent_dim.
            shared_codebook: VectorQuantization codebook C_f for the commonality level.
            text_codebook: VectorQuantization codebook C_t for the text-specific level.
            image_codebook: VectorQuantization codebook C_i for the image-specific level.
            text_decoder: D_t — reconstructs x_t from (e_f1 + e_t2).
            image_decoder: D_i — reconstructs x_i from (e_f1 + e_i3).
            alpha: Weight for the auxiliary InfoNCE loss.
            beta: Weight for the VQ + commitment loss.
            init_buffer_size: Number of points to buffer before initializing each codebook.
            optimizer: Partial optimizer constructor; receives model parameters at training start.
            scheduler: Partial LR scheduler constructor.
            embedding_dim: Dimensionality of the raw text/image embeddings fed into
                the model. Used to build per-modality LayerNorm modules applied
                after L2 normalisation before the encoders.
            training_loop_function: Optional custom training loop (e.g. for DDP-aware
                K-means init). When provided, automatic_optimization is disabled and this
                function is responsible for calling optimizer.step(). Signature:
                fn(module, loss, world_size, is_initialized).
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "text_encoder", "image_encoder", "fusion_layer",
                "text_fusion_mlp", "image_fusion_mlp",
                "shared_codebook", "text_codebook", "image_codebook",
                "text_decoder", "image_decoder",
                "optimizer", "scheduler", "training_loop_function",
            ],
        )

        # L2 normalisation + LayerNorm applied to raw embeddings before encoding.
        # L2 norm removes magnitude variation; LayerNorm re-scales to a learned
        # distribution, giving encoders a stable, unit-sphere-anchored input.
        self.text_input_norm = nn.LayerNorm(embedding_dim)
        self.image_input_norm = nn.LayerNorm(embedding_dim)

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_layer = fusion_layer
        self.text_fusion_mlp = text_fusion_mlp
        self.image_fusion_mlp = image_fusion_mlp
        self.shared_codebook = shared_codebook
        self.text_codebook = text_codebook
        self.image_codebook = image_codebook
        self.text_decoder = text_decoder
        self.image_decoder = image_decoder

        self.alpha = alpha
        self.beta = beta

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_loop_function = training_loop_function
        if self.training_loop_function is not None:
            self.automatic_optimization = False

        # Initialise at log(0.07) following CLIP; clamped during forward to stay positive.
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

        self.init_buffer_size = init_buffer_size
        for cb in (self.shared_codebook, self.text_codebook, self.image_codebook):
            cb.init_buffer_size = init_buffer_size

        self.train_loss = MeanMetric()
        self.train_rec_loss = MeanMetric()
        self.train_aux_loss = MeanMetric()
        self.train_vq_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_rec_loss = MeanMetric()
        self.val_aux_loss = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_rec_loss = MeanMetric()
        self.test_aux_loss = MeanMetric()

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (id_shared, id_text, id_img)
        Tuple[torch.Tensor, torch.Tensor],                 # (x_text_hat, x_img_hat)
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (z_f, z_t, z_i)
        torch.Tensor,                                      # vq_loss
    ]:
        """
        Args:
            text_embedding:  raw text feature  (B, text_dim)
            image_embedding: raw image feature (B, img_dim)

        Returns:
            ids:          (id_shared, id_text, id_img) each of shape (B,)
            recons:       (x_text_hat, x_img_hat)
            latents:      (z_f, z_t, z_i) for loss computation
            vq_loss:      scalar; zero during eval (no codebook gradient needed)
        """
        x_t = self.text_input_norm(F.normalize(text_embedding, dim=-1))
        x_i = self.image_input_norm(F.normalize(image_embedding, dim=-1))
        z_t = self.text_encoder(x_t)
        z_i = self.image_encoder(x_i)
        z_f = self.fusion_layer(z_t, z_i)

        is_training = (
            hasattr(self, "trainer") and self.trainer.state.fn == TrainerFn.FITTING
        )

        # ----- Level 1: shared codebook -----
        if is_training:
            id_shared, e_f1_ste, vq_loss_f = self.shared_codebook.model_step(z_f)
        else:
            id_shared, e_f1_ste = self.shared_codebook.predict_step(z_f)
            vq_loss_f = torch.tensor(0.0, device=self.device)

        # r_f = z_f - e_f1 (hard codebook entry, no STE).
        # Detaching e_f1_ste collapses the STE back to the pure codebook vector.
        # Gradient on z_f is kept so that downstream commitment terms for the
        # text/image paths further regularise z_f.
        r_f = z_f - e_f1_ste.detach()

        # ----- Level 2: text-specific codebook -----
        v_t = self.text_fusion_mlp(torch.cat([r_f, z_t], dim=-1))
        if is_training:
            id_text, e_t2_ste, vq_loss_t = self.text_codebook.model_step(v_t)
        else:
            id_text, e_t2_ste = self.text_codebook.predict_step(v_t)
            vq_loss_t = torch.tensor(0.0, device=self.device)

        # ----- Level 3: image-specific codebook -----
        v_i = self.image_fusion_mlp(torch.cat([r_f, z_i], dim=-1))
        if is_training:
            id_img, e_i3_ste, vq_loss_i = self.image_codebook.model_step(v_i)
        else:
            id_img, e_i3_ste = self.image_codebook.predict_step(v_i)
            vq_loss_i = torch.tensor(0.0, device=self.device)

        # Dropout on the modality-specific codes forces the fusion code e_f1 to
        # carry enough information to reconstruct on its own, preventing collapse
        # onto either the text or image path alone.
        x_text_hat = self.text_decoder(e_f1_ste + e_t2_ste)
        x_img_hat = self.image_decoder(e_f1_ste + e_i3_ste)

        vq_loss = vq_loss_f + vq_loss_t + vq_loss_i

        return (
            (id_shared, id_text, id_img),
            (x_text_hat, x_img_hat),
            (z_f, z_t, z_i),
            vq_loss,
        )

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _info_nce(
        self, anchor: torch.Tensor, positive: torch.Tensor
    ) -> torch.Tensor:
        """In-batch InfoNCE with learnable temperature (symmetric negatives)."""
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        temp = self.log_temperature.exp().clamp(min=1e-4)
        logits = torch.matmul(a, p.T) / temp   # (B, B): diagonal is the positive pair
        labels = torch.arange(a.shape[0], device=a.device)
        return F.cross_entropy(logits, labels)

    @property
    def _all_codebooks_initialized(self) -> bool:
        return (
            self.shared_codebook.is_initialized
            and self.text_codebook.is_initialized
            and self.image_codebook.is_initialized
        )

    # ------------------------------------------------------------------
    # model_step (shared across train / val / test)
    # ------------------------------------------------------------------

    def model_step(
        self, batch: ItemData
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns:
            ids:        (id_shared, id_text, id_img)
            total_loss: L_total scalar
            rec_loss:   L_rec  scalar
            aux_loss:   L_aux  scalar
            vq_loss:    L_vq   scalar
        """
        text_emb = batch.transformed_features["text_embedding"].to(self.device)
        img_emb = batch.transformed_features["image_embedding"].to(self.device)

        ids, (x_text_hat, x_img_hat), (z_f, z_t, z_i), vq_loss = self.forward(
            text_emb, img_emb
        )

        # Reconstruction loss is deferred until all codebooks are warmed up so that
        # early zero embeddings (during K-means init) do not corrupt the decoders.
        if self._all_codebooks_initialized:
            rec_loss = F.mse_loss(x_text_hat, text_emb) + F.mse_loss(x_img_hat, img_emb)
        else:
            rec_loss = torch.tensor(0.0, device=self.device)

        # InfoNCE is computed from the very first step to anchor z_f early.
        aux_loss = self._info_nce(z_f, z_t) + self._info_nce(z_f, z_i)

        total_loss = rec_loss + self.alpha * aux_loss + self.beta * vq_loss

        return ids, total_loss, rec_loss, aux_loss, vq_loss

    # ------------------------------------------------------------------
    # Lightning hooks: training
    # ------------------------------------------------------------------

    def training_step(self, batch: Tuple[ItemData], batch_idx: int) -> torch.Tensor:
        model_input: ItemData = batch[0]
        ids, total_loss, rec_loss, aux_loss, vq_loss = self.model_step(model_input)

        self.train_loss(total_loss)
        self.train_rec_loss(rec_loss)
        self.train_aux_loss(aux_loss)
        self.train_vq_loss(vq_loss)
        self.log_dict(
            {
                "train/loss": self.train_loss,
                "train/rec_loss": self.train_rec_loss,
                "train/aux_loss": self.train_aux_loss,
                "train/vq_loss": self.train_vq_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.training_loop_function is not None:
            self.training_loop_function(
                self,
                loss=total_loss,
                world_size=self.trainer.world_size,
                is_initialized=self._all_codebooks_initialized,
            )

        return total_loss

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.train_rec_loss.reset()
        self.train_aux_loss.reset()
        self.train_vq_loss.reset()
        for cb in (self.shared_codebook, self.text_codebook, self.image_codebook):
            cb.on_train_start()

    # ------------------------------------------------------------------
    # Lightning hooks: validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: ItemData, batch_idx: int) -> None:
        _, total_loss, rec_loss, aux_loss, _ = self.model_step(batch)
        self.val_loss(total_loss)
        self.val_rec_loss(rec_loss)
        self.val_aux_loss(aux_loss)
        self.log_dict(
            {
                "val/loss": self.val_loss,
                "val/rec_loss": self.val_rec_loss,
                "val/aux_loss": self.val_aux_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_start(self) -> None:
        self.val_loss.reset()
        self.val_rec_loss.reset()
        self.val_aux_loss.reset()

    # ------------------------------------------------------------------
    # Lightning hooks: test
    # ------------------------------------------------------------------

    def test_step(self, batch: ItemData, batch_idx: int) -> None:
        _, total_loss, rec_loss, aux_loss, _ = self.model_step(batch)
        self.test_loss(total_loss)
        self.test_rec_loss(rec_loss)
        self.test_aux_loss(aux_loss)
        self.log_dict(
            {
                "test/loss": self.test_loss,
                "test/rec_loss": self.test_rec_loss,
                "test/aux_loss": self.test_aux_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        self.test_loss.reset()
        self.test_rec_loss.reset()
        self.test_aux_loss.reset()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_step(self, batch: ItemData) -> OneKeyPerPredictionOutput:
        """Returns the three-level semantic IDs for each item in the batch."""
        text_emb = batch.transformed_features["text_embedding"].to(self.device)
        img_emb = batch.transformed_features["image_embedding"].to(self.device)

        (id_shared, id_text, id_img), _, _, _ = self.forward(text_emb, img_emb)
        cluster_ids = torch.stack([id_shared, id_text, id_img], dim=-1)  # (B, 3)

        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]
        return OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.optimizer is not None:
            optimizer = self.optimizer(params=self.trainer.model.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["codebooks_initialized"] = {
            "shared": self.shared_codebook.is_initialized,
            "text": self.text_codebook.is_initialized,
            "image": self.image_codebook.is_initialized,
        }
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        states = checkpoint.get("codebooks_initialized", {})
        self.shared_codebook.is_initialized = states.get("shared", False)
        self.text_codebook.is_initialized = states.get("text", False)
        self.image_codebook.is_initialized = states.get("image", False)
        return super().on_load_checkpoint(checkpoint)

    def log_if_true(self, message: str, condition: bool) -> None:
        if condition:
            logging.info(f"Device {self.device}: {message}")