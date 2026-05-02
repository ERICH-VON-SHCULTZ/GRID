import functools
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError
from src.components.quantization_strategies import QuantizationStrategy
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule

log = logging.getLogger(__name__)


class VectorQuantization(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
    ):
        """
        Initialize the VectorQuantization module.

        Args:
            n_clusters: Number of clusters.
            n_features: Number of features in the input data.
            distance_function: Distance function to use for computing distances between points.
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for training.
            init_method: Initialization method ("random" or "k-means++").
            init_buffer_size: Number of points to buffer for initialization.
        """

        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            loss_function=loss_function,
            optimizer=optimizer,
            initializer=initializer,
            init_buffer_size=init_buffer_size,
        )

        self.quantization_strategy = quantization_strategy

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the K-Means model on the input batch.

        This function computes the cluster assignments for each input point, the number
        of points in each cluster, and the sum of points in each cluster.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            embeddings: Embeddings of shape (batch_size, n_features).
                These embeddings will be used for computing the quantization loss.
            reconstruction_loss_embeddings: Embeddings of shape (batch_size, n_features)
                computed in a way that enables gradient backpropagation through the input
                embeddings. If the quantization strategy does not support this, this will
                be None.
        """
        codebook = self.get_centroids()
        (
            ids,
            embeddings,
            reconstruction_loss_embeddings,
        ) = self.quantization_strategy.quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Perform a forward pass of the K-Means model on the batch and compute the loss.

        This function may be called by another LightningModule, such as a residual
        K-means module, that is using this MiniBatchKMeans module as a submodule.

        Calling this function along will not update the centroids, and will not
        increment self.global_step. If a parent module is using this module as a
        submodule, the parent will be responsible for updating those parameters.
        Otherwise, these will be updated by Lightning after it calls
        training_step.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            global_loss_embeddings: Embeddings of shape (batch_size, n_features)
            loss: Loss value. Tensor of shape (1,)
        """
        if batch.device != self.device:
            batch = batch.to(self.device)

        # Initialize centroids using the chosen method
        # Buffer initial batches for better initialization
        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, embeddings, reconstruction_loss_embeddings = self.forward(batch)
        loss = self.loss_function(batch, embeddings)  # quantization loss
        return (
            assignments,
            reconstruction_loss_embeddings
            if reconstruction_loss_embeddings is not None
            else embeddings,
            loss,
        )


class EMAVectorQuantization(VectorQuantization):
    """VectorQuantization with EMA codebook updates and dead code revival.

    The codebook is NOT updated by gradient descent. Instead, each training step
    updates it via exponential moving averages:
      N_k <- gamma * N_k + (1-gamma) * n_k
      m_k <- gamma * m_k + (1-gamma) * sum(z_i : assign(z_i) = k)
      e_k  = m_k / N_k   (Laplace-smoothed)

    Only the commitment loss (encoder pulled toward codebook) is backpropagated.
    Dead codes whose EMA cluster size falls below dead_code_threshold are revived
    by replacing them with randomly sampled inputs from the current batch.
    """

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: Optional[torch.nn.Module] = None,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        init_buffer_size: int = 1000,
    ):
        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            initializer=initializer,
            quantization_strategy=quantization_strategy,
            loss_function=loss_function if loss_function is not None else WeightedSquaredError(),
            optimizer=None,
            init_buffer_size=init_buffer_size,
        )
        # Codebook is owned by EMA, not the optimizer.
        self.centroids.requires_grad_(False)

        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # EMA accumulators. Seeded at 1 / zeros so the first real batch fully
        # adopts its statistics without being diluted by a cold start.
        self.register_buffer("ema_cluster_size", torch.ones(n_clusters))
        self.register_buffer("ema_embed_sum", torch.zeros(n_clusters, n_features))

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _ema_update(self, batch: torch.Tensor, assignments: torch.Tensor) -> None:
        one_hot = torch.zeros(batch.shape[0], self.n_clusters, device=batch.device)
        one_hot.scatter_(1, assignments.unsqueeze(1), 1.0)

        n_k = one_hot.sum(0)           # (K,)
        embed_sum = one_hot.T @ batch  # (K, d)

        # Sync across DDP workers so every replica updates identically.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(n_k)
            torch.distributed.all_reduce(embed_sum)

        self.ema_cluster_size.mul_(self.ema_decay).add_(n_k, alpha=1.0 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)

        # Laplace smoothing to avoid division by zero for empty clusters.
        N = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (N + self.n_clusters * 1e-5) * N
        self.centroids.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    def _revive_dead_codes(self, batch: torch.Tensor) -> None:
        dead = self.ema_cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        idx = torch.randint(0, batch.shape[0], (n_dead,), device=batch.device)
        self.centroids.data[dead] = batch[idx]
        self.ema_cluster_size[dead] = self.dead_code_threshold
        self.ema_embed_sum[dead] = batch[idx]
        log.debug(f"Revived {n_dead} dead codes.")

    # ------------------------------------------------------------------
    # Override initialization to set EMA state from K-means++ centroids
    # and broadcast across DDP ranks.
    # ------------------------------------------------------------------

    def initialization_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Buffer K-means++ inputs in float32 regardless of AMP precision.
        self._buffer_points(batch.float())
        B = batch.shape[0]
        zero_emb = torch.zeros_like(batch)
        zero_ids = torch.zeros(B, dtype=torch.long, device=self.device)
        zero_loss = torch.tensor(0.0, device=self.device)

        if self.init_buffer.shape[0] < self.init_buffer_size:
            return zero_ids, zero_emb, zero_loss

        # Buffer full — run K-means++ on rank 0, then broadcast.
        self.init_centroids = torch.zeros_like(self.centroids.data)
        self.compute_initial_centroids(self.init_buffer)  # rank-zero-only

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.init_centroids, src=0)

        self.centroids.data.copy_(self.init_centroids)
        self.ema_embed_sum.copy_(self.init_centroids)
        self.ema_cluster_size.fill_(1.0)

        self.is_initial_step = True
        self.init_buffer = torch.tensor([], device=self.device)

        distances = self.distance_function.compute(batch, self.centroids.data)
        assignments = torch.argmin(distances, dim=1)
        return assignments, self.centroids[assignments], zero_loss

    # ------------------------------------------------------------------
    # Override model_step: EMA update + commitment-only loss
    # ------------------------------------------------------------------

    def model_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.device != self.device:
            batch = batch.to(self.device)

        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        # Cast to float32 for EMA buffers; batch may be bf16 under mixed precision.
        batch_f32 = batch.detach().to(self.centroids.dtype)

        if self.training:
            self._revive_dead_codes(batch_f32)

        assignments, embeddings, recon_embeddings = self.forward(batch)

        if self.training:
            self._ema_update(batch_f32, assignments.detach())

        # Commitment loss only: pull encoder toward codebook; no codebook gradient.
        commitment_loss = F.mse_loss(batch, embeddings.detach())

        return (
            assignments,
            recon_embeddings if recon_embeddings is not None else embeddings,
            commitment_loss,
        )
