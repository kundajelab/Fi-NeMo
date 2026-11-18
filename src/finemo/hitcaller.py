"""Hit caller module implementing the Fi-NeMo motif instance calling algorithm.

This module provides the core functionality for identifying transcription factor
binding motif instances in neural network contribution scores using a competitive
optimization approach based on proximal gradient descent.

The main algorithm fits a sparse linear model where contribution scores are
reconstructed as a weighted combination of motif contribution weight matrices (CWMs)
at specific genomic positions. The sparsity constraint ensures that only the most
significant motif instances are called.
"""

import contextlib
import warnings
from typing import Tuple, Union, Optional, Dict, List
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
from torch import Tensor
import polars as pl
from jaxtyping import Float, Int, Bool

from tqdm import tqdm


def prox_grad_step(
    coefficients: Float[Tensor, "B M P"],
    importance_scale: Float[Tensor, "B 1 P"],
    cwms: Float[Tensor, "M 4 W"],
    contribs: Float[Tensor, "B 4 L"],
    sequences: Union[Int[Tensor, "B 4 L"], int],
    lambdas: Float[Tensor, "1 M 1"],
    step_sizes: Float[Tensor, "B 1 1"],
    compute_gap: bool = True,
) -> Tuple[Float[Tensor, "B M P"], Optional[Tensor], Optional[Tensor]]:
    """Perform a proximal gradient descent optimization step for non-negative lasso.

    This function implements a single optimization step of the Fi-NeMo algorithm,
    which uses proximal gradient descent to solve a sparse reconstruction problem.
    The goal is to represent contribution scores as a sparse linear combination
    of motif contribution weight matrices (CWMs).

    Dimension notation:
    - B = batch size (number of regions processed simultaneously)
    - M = number of motifs
    - L = sequence length
    - W = motif width (length of each motif)
    - P = L - W + 1 (number of valid motif positions)

    Parameters
    ----------
    coefficients : Float[Tensor, "B M P"]
        Current coefficient matrix representing motif instance strengths.
    importance_scale : Float[Tensor, "B 1 P"]
        Scaling factors for importance-weighted reconstruction.
    cwms : Float[Tensor, "M 4 W"]
        Motif contribution weight matrices for all motifs.
        4 represents the DNA bases (A, C, G, T).
    contribs : Float[Tensor, "B 4 L"]
        Target contribution scores to reconstruct.
    sequences : Float[Tensor, "B 4 L"] | int
        One-hot encoded DNA sequences. Can be a scalar (1) for hypothetical mode.
    lambdas : Float[Tensor, "1 M 1"]
        L1 regularization weights for each motif.
    step_sizes : Float[Tensor, "B 1 1"]
        Optimization step sizes for each batch element.
    compute_gap : bool, default True
        Whether to compute the duality gap and NLL of the new coefficients, which costs a
        second pair of convolutions. Without it both are returned as None.

    Returns
    -------
    c_next : Float[Tensor, "B M P"]
        Updated coefficient matrix after the optimization step (shape: batch_size × motifs × positions).
    dual_gap : Float[Tensor, " B"] or None
        Duality gap of c_next for convergence assessment (shape: batch_size).
    nll : Float[Tensor, " B"] or None
        Squared residual norm of c_next (twice the NLL term of the objective, shape: batch_size).

    Notes
    -----
    The algorithm uses proximal gradient descent to solve:

    minimize_c: ||contribs - conv_transpose(c * importance_scale, cwms) * sequences||²₂ + λ||c||₁

    subject to: c ≥ 0

    References
    ----------
    - Proximal gradient descent: https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slide 22
    - Duality gap computation: https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf, Section III
    """
    # Forward pass: convolution operations require specific tensor layouts
    coef_adj = coefficients * importance_scale
    pred_unmasked = F.conv_transpose1d(coef_adj, cwms)  # (B, 4, L)
    pred = (
        pred_unmasked * sequences
    )  # (B, 4, L), element-wise masking for projected mode

    # Compute gradient * -1
    residuals = contribs - pred  # (B, 4, L)
    ngrad = F.conv1d(residuals, cwms) * importance_scale  # (B, M, P)

    # Compute proximal gradient descent step
    c_next = coefficients + step_sizes * (ngrad - lambdas)  # (B, M, P)
    c_next = F.relu(c_next)  # Ensure non-negativity constraint
    if not compute_gap:
        return c_next, None, None

    # Duality gap at the new, feasible coefficients (the ones that are saved), with the
    # dual point t * r, t = 1 / max(1, max_j (A'r)_j / lambda_j). The gap
    # 1/2 (1 + t^2) |r|^2 - t r'y + lambda'c is written as
    # 1/2 (1 - t)^2 |r|^2 + sum_j c_j (lambda_j - t (A'r)_j): both terms are >= 0, so
    # nothing cancels in float32. Clamping before the reciprocal keeps t = 1 when every
    # correlation is negative.
    r_next = contribs - F.conv_transpose1d(c_next * importance_scale, cwms) * sequences
    g_next = F.conv1d(r_next, cwms) * importance_scale  # (B, M, P)
    t = 1 / (g_next / lambdas).amax(dim=(1, 2)).clamp(min=1.0)  # (B)
    nll = (r_next**2).sum(dim=(1, 2))  # (B), residual of the saved coefficients
    dual_gap = 0.5 * (1 - t) ** 2 * nll + (
        c_next * (lambdas - t[:, None, None] * g_next)
    ).sum(dim=(1, 2))  # (B)

    return c_next, dual_gap, nll


def optimizer_step(
    cwms: Float[Tensor, "M 4 W"],
    contribs: Float[Tensor, "B 4 L"],
    importance_scale: Float[Tensor, "B 1 P"],
    sequences: Union[Int[Tensor, "B 4 L"], int],
    coef_inter: Float[Tensor, "B M P"],
    coef: Float[Tensor, "B M P"],
    i: Float[Tensor, "B 1 1"],
    step_sizes: Float[Tensor, "B 1 1"],
    L: int,
    lambdas: Float[Tensor, "1 M 1"],
    compute_gap: bool = True,
) -> Tuple[
    Float[Tensor, "B M P"],
    Float[Tensor, "B M P"],
    Optional[Tensor],
    Optional[Tensor],
]:
    """Perform a non-negative lasso optimizer step with Nesterov momentum.

    This function combines proximal gradient descent with momentum acceleration
    to improve convergence speed while maintaining the non-negative constraint
    on coefficients.

    Dimension notation:
    - B = batch size (number of regions processed simultaneously)
    - M = number of motifs
    - L = sequence length
    - W = motif width (length of each motif)
    - P = L - W + 1 (number of valid motif positions)

    Parameters
    ----------
    cwms : Float[Tensor, "M 4 W"]
        Motif contribution weight matrices.
    contribs : Float[Tensor, "B 4 L"]
        Target contribution scores.
    importance_scale : Float[Tensor, "B 1 P"]
        Importance scaling factors.
    sequences : Union[Int[Tensor, "B 4 L"], int]
        One-hot encoded sequences or scalar for hypothetical mode.
    coef_inter : Float[Tensor, "B M P"]
        Intermediate coefficient matrix (with momentum).
    coef : Float[Tensor, "B M P"]
        Current coefficient matrix.
    i : Float[Tensor, "B 1 1"]
        Iteration counter for each batch element.
    step_sizes :  Float[Tensor, "B 1 1"]
        Step sizes for optimization.
    L : int
        Sequence length for normalization.
    lambdas : Float[Tensor, "1 M 1"]
        Regularization parameters.
    compute_gap : bool, default True
        Whether to compute the duality gap and NLL (None otherwise); see prox_grad_step.

    Returns
    -------
    coef_inter : Float[Tensor, "B M P"]
        Updated intermediate coefficients with momentum (shape: batch_size × motifs × positions).
    coef : Float[Tensor, "B M P"]
        Updated coefficient matrix (shape: batch_size × motifs × positions).
    gap : Float[Tensor, " B"] or None
        Normalized duality gap (shape: batch_size).
    nll : Float[Tensor, " B"] or None
        Normalized negative log likelihood (shape: batch_size).

    Notes
    -----
    Uses Nesterov momentum with momentum coefficient i/(i+3) for improved
    convergence properties. The duality gap and NLL are normalized by
    sequence length for scale-invariant convergence assessment.

    References
    ----------
    https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf, slides 22, 27
    """
    coef_prev = coef

    # Proximal gradient descent step
    coef, gap, nll = prox_grad_step(
        coef_inter, importance_scale, cwms, contribs, sequences, lambdas, step_sizes,
        compute_gap,
    )
    if gap is not None:
        gap = gap / L
        nll = nll / (2 * L)

    # Compute updated coefficients with Nesterov momentum
    mom_term = i / (i + 3.0)
    coef_inter = (1 + mom_term) * coef - mom_term * coef_prev

    return coef_inter, coef, gap, nll


def _to_channel_last_layout(tensor: Tensor, **kwargs) -> torch.Tensor:
    """Convert tensor to channel-last memory layout for optimized convolution operations.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to convert.
    **kwargs
        Additional keyword arguments passed to tensor.to().

    Returns
    -------
    torch.Tensor
        Tensor with channel-last memory layout.
    """
    return (
        tensor[:, :, :, None].to(memory_format=torch.channels_last, **kwargs).squeeze(3)
    )


@contextlib.contextmanager
def _tf32_setting(allow_tf32: Optional[bool]):
    """Set whether convolutions (and matrix products) may use TF32 inside the block; None leaves PyTorch's setting."""
    if allow_tf32 is None:
        yield
        return
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Apply signed square root transformation to input tensor.

    This transformation preserves the sign while applying square root to the
    absolute value, which can help with numerical stability and gradient flow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Transformed tensor with same shape as input.
    """
    return torch.sign(x) * torch.sqrt(torch.abs(x))


class BatchLoaderBase(ABC):
    """Base class for loading batches of contribution scores and sequences.

    This class provides common functionality for different input formats
    including batch indexing and padding for consistent batch sizes.

    Dimension notation:
    - N = number of sequences/regions in dataset
    - L = sequence length
    - B = batch size (number of regions processed simultaneously)

    Parameters
    ----------
    contribs : Union[Float[Tensor, "N 4 L"], Float[Tensor, "N L"]]
        Contribution scores array.
    sequences : Int[Tensor, "N 4 L"]
        One-hot encoded sequences array.
    L : int
        Sequence length.
    device : torch.device
        Target device for tensor operations.
    """

    def __init__(
        self,
        contribs: Union[Float[Tensor, "N 4 L"], Float[Tensor, "N L"]],
        sequences: Int[Tensor, "N 4 L"],
        L: int,
        device: torch.device,
    ) -> None:
        self.contribs = contribs
        self.sequences = sequences
        self.L = L
        self.device = device

    def _get_inds_and_pad_lens(
        self, start: int, end: int
    ) -> Tuple[Int[Tensor, " Z"], Tuple[int, ...]]:
        """Get indices and padding lengths for batch loading.

        Parameters
        ----------
        start : int
            Start index for batch.
        end : int
            End index for batch.

        Returns
        -------
        inds : Int[Tensor, " Z"]
            Padded indices tensor with -1 for padding positions (shape: padded_batch_size).
        pad_lens : tuple
            Padding specification for F.pad (left, right, top, bottom, front, back).
        """
        N = end - start
        end = min(end, self.contribs.shape[0])
        overhang = N - (end - start)
        pad_lens = (0, 0, 0, 0, 0, overhang)

        inds = F.pad(
            torch.arange(start, end, dtype=torch.int), (0, overhang), value=-1
        ).to(device=self.device)

        return inds, pad_lens

    @abstractmethod
    def load_batch(
        self, start: int, end: int
    ) -> Tuple[
        Float[Tensor, "B 4 L"], Union[Int[Tensor, "B 4 L"], int], Int[Tensor, " B"]
    ]:
        """Load a batch of data.

        Dimension notation:
        - B = batch size (number of regions in this batch)
        - L = sequence length

        Parameters
        ----------
        start : int
            Start index (used by subclasses).
        end : int
            End index (used by subclasses).

        Returns
        -------
        contribs_batch : Float[Tensor, "B 4 L"]
            Batch of contribution scores (shape: batch_size × 4_bases × L).
        sequences_batch : Union[Int[Tensor, "B 4 L"], int]
            Batch of one-hot encoded sequences (shape: batch_size × 4_bases × L) or scalar 1 for hypothetical mode.
        inds_batch : Int[Tensor, " B"]
            Batch indices mapping to original sequence indices (shape: batch_size).

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Parameters are intentionally unused in the base implementation.
        """
        pass


class BatchLoaderCompactFmt(BatchLoaderBase):
    """Batch loader for compact format contribution scores.

    Handles contribution scores in shape (N, L) representing projected
    scores that need to be broadcasted to (N, 4, L) format.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], Int[Tensor, "B 4 L"], Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_compact = F.pad(self.contribs[start:end, None, :], pad_lens)
        contribs_batch = _to_channel_last_layout(
            contribs_compact, device=self.device, dtype=torch.float32
        )
        sequences_batch = F.pad(self.sequences[start:end, :, :], pad_lens)  # (B, 4, L)
        sequences_batch = _to_channel_last_layout(
            sequences_batch, device=self.device, dtype=torch.int8
        )

        contribs_batch = contribs_batch * sequences_batch  # (B, 4, L)

        return contribs_batch, sequences_batch, inds


class BatchLoaderProj(BatchLoaderBase):
    """Batch loader for projected contribution scores.

    Handles contribution scores in shape (N, 4, L) where scores are
    element-wise multiplied by one-hot sequences to get projected contributions.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], Int[Tensor, "B 4 L"], Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_hyp = F.pad(self.contribs[start:end, :, :], pad_lens)
        contribs_hyp = _to_channel_last_layout(
            contribs_hyp, device=self.device, dtype=torch.float32
        )
        sequences_batch = F.pad(self.sequences[start:end, :, :], pad_lens)  # (B, 4, L)
        sequences_batch = _to_channel_last_layout(
            sequences_batch, device=self.device, dtype=torch.int8
        )
        contribs_batch = contribs_hyp * sequences_batch

        return contribs_batch, sequences_batch, inds


class BatchLoaderHyp(BatchLoaderBase):
    """Batch loader for hypothetical contribution scores.

    Handles hypothetical contribution scores in shape (N, 4, L) where
    scores represent counterfactual effects of base substitutions.
    """

    def load_batch(
        self, start: int, end: int
    ) -> Tuple[Float[Tensor, "B 4 L"], int, Int[Tensor, " B"]]:
        inds, pad_lens = self._get_inds_and_pad_lens(start, end)

        contribs_batch = F.pad(self.contribs[start:end, :, :], pad_lens)
        contribs_batch = _to_channel_last_layout(
            contribs_batch, device=self.device, dtype=torch.float32
        )

        return contribs_batch, 1, inds


def fit_contribs(
    cwms: Float[ndarray, "M 4 W"],
    contribs: Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]],
    sequences: Int[ndarray, "N 4 L"],
    cwm_trim_mask: Float[ndarray, "M W"],
    use_hypothetical: bool,
    lambdas: Float[ndarray, " M"],
    step_size_max: float = 3.0,
    step_size_min: float = 0.08,
    sqrt_transform: bool = False,
    convergence_tol: float = 0.00005,
    max_steps: int = 10000,
    batch_size: int = 2000,
    step_adjust: float = 0.7,
    post_filter: bool = True,
    device: Optional[torch.device] = None,
    compile_optimizer: bool = False,
    eps: float = 1.0,
    stall_start: int = 2000,
    stall_window: int = 1000,
    stall_improvement: float = 0.01,
    early_restart: bool = True,
    check_every: int = 5,
    allow_tf32: Optional[bool] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Call motif hits by fitting sparse linear model to contribution scores.

    This is the main function implementing the Fi-NeMo algorithm. It identifies
    motif instances by solving a sparse reconstruction problem where contribution
    scores are approximated as a linear combination of motif CWMs at specific
    positions. The optimization uses proximal gradient descent with momentum.

    Parameters
    ----------
    cwms : Float[ndarray, "M 4 W"]
        Motif contribution weight matrices where:
        - M = number of motifs (transcription factor binding patterns)
        - 4 = DNA bases (A, C, G, T dimensions)
        - W = motif width (length of each motif pattern)
    contribs : Float[ndarray, "N 4 L"] | Float[ndarray, "N L"]
        Neural network contribution scores where:
        - N = number of regions in dataset
        - L = sequence length
        Can be hypothetical (N, 4, L) or projected (N, L) format.
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences (shape: num_regions × 4_bases × L).
    cwm_trim_mask : Float[ndarray, "M W"]
        Binary mask indicating which positions of each CWM to use (shape: num_motifs × motif_width).
    use_hypothetical : bool
        Whether to use hypothetical contribution scores (True) or
        projected scores (False).
    lambdas : Float[ndarray, " M"]
        L1 regularization weights for each motif (shape: num_motifs).
    step_size_max : float, default 3.0
        Maximum optimization step size.
    step_size_min : float, default 0.08
        Minimum optimization step size (for convergence failure detection).
    sqrt_transform : bool, default False
        Whether to apply signed square root transformation to inputs.
    convergence_tol : float, default 0.00005
        Convergence tolerance based on duality gap.
    max_steps : int, default 10000
        Maximum number of optimization steps per region, restarts included.
    batch_size : int, default 2000
        Number of regions to process simultaneously.
    step_adjust : float, default 0.7
        Factor to reduce step size when optimization diverges.
    post_filter : bool, default True
        Whether to filter hits based on similarity threshold.
    device : torch.device, optional
        Target device for computation. Auto-detected if None.
    compile_optimizer : bool, default False
        Whether to JIT compile the optimizer for speed.
    eps : float, default 1.0
        Small constant for numerical stability.
    stall_start, stall_window, stall_improvement : int, int, float, default 2000, 1000, 0.01
        Stall safeguard. The fixed-step momentum iteration can cycle in a few regions without
        converging or diverging. After ``stall_start`` steps, at the end of every ``stall_window``
        steps, a region whose best duality gap over the window improves by less than the fraction
        ``stall_improvement`` on the window before has its step size multiplied by ``step_adjust``
        and its momentum reset.
    early_restart : bool, default True
        Restart a region with a smaller step as soon as its objective exceeds that of the empty
        fit (0.5 per bp, since the scores are RMS-normalized) by 0.1%, instead of waiting for the
        divergence to overflow to a non-finite gap.
    check_every : int, default 5
        Compute the duality gap and assess convergence, divergence, stalls and timeouts only
        every ``check_every`` steps. The steps in between skip the gap's extra convolutions and
        the transfers from the device (about half the cost of a step). A region stops only at a
        check, so it may run past a momentary dip of its gap below the tolerance; on DNase/ATAC
        regions 5 took 21% more steps but 1.5 times less time than 1. ``stall_start`` and
        ``stall_window`` must be multiples of it.
    allow_tf32 : bool, optional
        Whether the convolutions may run in TF32 on GPUs that support it (Ampere and newer).
        TF32 is faster but rounds the inputs of every convolution to a 10-bit mantissa, so
        results differ slightly between GPU types, and the duality gap is computed less exactly:
        in emulation at a tolerance of 5e-5 per bp, 12-18% of DNase/ATAC regions stopped with a
        true gap above the tolerance (at most 1.5 times it), against none in FP32. Use False for
        an exact certificate and results that agree across hardware. None (the default) keeps
        PyTorch's setting (``torch.backends.cudnn.allow_tf32``, True unless changed); the
        setting is restored afterwards.

    Returns
    -------
    hits_df : pl.DataFrame
        DataFrame containing called motif hits with columns:
        - peak_id: Region index
        - motif_id: Motif index
        - hit_start: Start position of hit
        - hit_coefficient: Hit strength coefficient
        - hit_similarity: Cosine similarity with motif
        - hit_importance: Total contribution score in hit region
        - hit_importance_sq: Sum of squared contributions (for normalization)
    qc_df : pl.DataFrame
        DataFrame containing quality control metrics with columns:
        - peak_id: Region index
        - nll: Final negative log likelihood
        - dual_gap: Final duality gap
        - num_steps: Number of optimization steps since the last restart
        - num_steps_total: Number of optimization steps, restarts included
        - num_restarts: Number of restarts after a divergence
        - num_step_reductions: Number of step size reductions by the stall safeguard
        - num_filtered: Number of placements removed by the similarity post-filter
        - step_size: Final step size
        - global_scale: Region-level scaling factor

    Notes
    -----
    The algorithm solves the optimization problem:

    minimize_c: ||contribs - Σⱼ convolve(c * scale, cwms[j]) * sequences||²₂ + Σⱼ λⱼ||c[:,j]||₁

    subject to: c ≥ 0

    where c[i,j] represents the strength of motif j at position i.

    The importance scaling balances reconstruction across different
    motifs and positions based on the local contribution magnitude.

    Examples
    --------
    >>> hits_df, qc_df = fit_contribs(
    ...     cwms=motif_cwms,
    ...     contribs=contrib_scores,
    ...     sequences=onehot_seqs,
    ...     cwm_trim_mask=trim_masks,
    ...     use_hypothetical=False,
    ...     lambdas=np.array([0.7, 0.8]),
    ...     step_size_max=3.0,
    ...     step_size_min=0.08,
    ...     sqrt_transform=False,
    ...     convergence_tol=0.0005,
    ...     max_steps=10000,
    ...     batch_size=1000,
    ...     step_adjust=0.7,
    ...     post_filter=True,
    ...     device=None,
    ...     compile_optimizer=False
    ... )
    """
    M, _, W = cwms.shape
    N, _, L = sequences.shape

    B = batch_size  # Using uppercase for consistency with dimension notation

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("No GPU available. Running on CPU.", RuntimeWarning)

    # Compile optimizer if requested (local binding: later calls are unaffected)
    step_fn = (
        torch.compile(optimizer_step, fullgraph=True)
        if compile_optimizer
        else optimizer_step
    )

    # Convert inputs to PyTorch tensors with proper device placement
    cwms_tensor: torch.Tensor = torch.from_numpy(cwms)
    contribs_tensor: torch.Tensor = torch.from_numpy(contribs)
    sequences_tensor: torch.Tensor = torch.from_numpy(sequences)
    cwm_trim_mask_tensor = torch.from_numpy(cwm_trim_mask)[:, None, :].repeat(1, 4, 1)
    lambdas_tensor: torch.Tensor = torch.from_numpy(lambdas)[None, :, None].to(
        device=device, dtype=torch.float32
    )

    # Convert to channel-last layout for optimized convolution operations
    cwms_tensor = _to_channel_last_layout(
        cwms_tensor, device=device, dtype=torch.float32
    )
    cwm_trim_mask_tensor = _to_channel_last_layout(
        cwm_trim_mask_tensor, device=device, dtype=torch.float32
    )
    cwms_tensor = cwms_tensor * cwm_trim_mask_tensor  # Apply trimming mask

    if sqrt_transform:
        cwms_tensor = _signed_sqrt(cwms_tensor)

    # Unit norm after trimming (and the square root): hit_similarity is then the cosine
    # similarity between the trimmed CWM and the contributions it covers, and lambda means
    # the same for every motif, however much of it was trimmed
    cwm_norm = (cwms_tensor**2).sum(dim=(1, 2)).sqrt()
    cwms_tensor = cwms_tensor / cwm_norm[:, None, None]

    # Initialize batch loader
    if len(contribs_tensor.shape) == 3:
        if use_hypothetical:
            batch_loader = BatchLoaderHyp(contribs_tensor, sequences_tensor, L, device)
        else:
            batch_loader = BatchLoaderProj(contribs_tensor, sequences_tensor, L, device)
    elif len(contribs_tensor.shape) == 2:
        if use_hypothetical:
            raise ValueError(
                "Input regions do not contain hypothetical contribution scores"
            )
        else:
            batch_loader = BatchLoaderCompactFmt(
                contribs_tensor, sequences_tensor, L, device
            )
    else:
        raise ValueError(
            f"Input contributions array is of incorrect shape {contribs_tensor.shape}"
        )

    # Initialize output container objects
    hit_idxs_lst: List[ndarray] = []
    coefficients_lst: List[ndarray] = []
    similarity_lst: List[ndarray] = []
    importance_lst: List[ndarray] = []
    importance_sq_lst: List[ndarray] = []
    qc_lsts: Dict[str, List[ndarray]] = {
        "nll": [],
        "dual_gap": [],
        "num_steps": [],
        "num_steps_total": [],
        "num_restarts": [],
        "num_step_reductions": [],
        "num_filtered": [],
        "step_size": [],
        "global_scale": [],
        "peak_id": [],
    }

    # Initialize buffers for optimizer
    coef_inter: Float[Tensor, "B M P"] = torch.zeros(
        (B, M, L - W + 1)
    )  # (B, M, P) where P = L - W + 1
    coef_inter = _to_channel_last_layout(coef_inter, device=device, dtype=torch.float32)
    coef: Float[Tensor, "B M P"] = torch.zeros_like(coef_inter)
    i: Float[Tensor, "B 1 1"] = torch.zeros((B, 1, 1), dtype=torch.int, device=device)
    # Steps since the region was loaded, restarts included (i restarts at 0 after each)
    n_total: Int[Tensor, " B"] = torch.zeros((B,), dtype=torch.int, device=device)
    n_restarts: Int[Tensor, " B"] = torch.zeros((B,), dtype=torch.int, device=device)
    # Stall safeguard: best gap in the current window and in the window before
    n_reductions: Int[Tensor, " B"] = torch.zeros((B,), dtype=torch.int, device=device)
    gap_min: Float[Tensor, " B"] = torch.full((B,), float("inf"), device=device)
    gap_ref: Float[Tensor, " B"] = torch.full((B,), float("inf"), device=device)
    step_sizes: Float[Tensor, "B 1 1"] = torch.full(
        (B, 1, 1), step_size_max, dtype=torch.float32, device=device
    )

    converged: Bool[Tensor, " B"] = torch.full(
        (B,), True, dtype=torch.bool, device=device
    )
    num_load = B

    contribs_buf: Float[Tensor, "B 4 L"] = torch.zeros((B, 4, L))
    contribs_buf = _to_channel_last_layout(
        contribs_buf, device=device, dtype=torch.float32
    )

    seqs_buf: Union[Int[Tensor, "B 4 L"], int]
    if use_hypothetical:
        seqs_buf = 1
    else:
        seqs_buf = torch.zeros((B, 4, L))
        seqs_buf = _to_channel_last_layout(seqs_buf, device=device, dtype=torch.int8)

    importance_scale_buf: Float[Tensor, "B M P"] = torch.zeros((B, M, L - W + 1))
    importance_scale_buf = _to_channel_last_layout(
        importance_scale_buf, device=device, dtype=torch.float32
    )

    inds_buf: Int[Tensor, " B"] = torch.zeros((B,), dtype=torch.int, device=device)
    global_scale_buf: Float[Tensor, " B"] = torch.zeros(
        (B,), dtype=torch.float, device=device
    )

    if check_every < 1 or stall_start % check_every or stall_window % check_every:
        raise ValueError(
            f"check_every ({check_every}) must be positive and divide stall_start "
            f"({stall_start}) and stall_window ({stall_window})"
        )
    step_count = 0  # regions are loaded only after a check, so their step counts stay aligned with it

    with _tf32_setting(allow_tf32), tqdm(disable=None, unit="regions", total=N, ncols=120) as pbar:
        num_complete = 0
        next_ind = 0
        while num_complete < N:
            # Retire converged peaks and fill buffer with new data
            if num_load > 0:
                load_start = next_ind
                load_end = load_start + num_load
                next_ind = min(load_end, contribs_tensor.shape[0])

                batch_data = batch_loader.load_batch(int(load_start), int(load_end))
                contribs_batch, seqs_batch, inds_batch = batch_data

                if sqrt_transform:
                    contribs_batch = _signed_sqrt(contribs_batch)

                global_scale_batch = ((contribs_batch**2).sum(dim=(1, 2)) / L).sqrt()
                contribs_batch = torch.nan_to_num(
                    contribs_batch / global_scale_batch[:, None, None]
                )

                importance_scale_batch = (
                    F.conv1d(contribs_batch**2, cwm_trim_mask_tensor) + eps
                ) ** (-0.5)
                importance_scale_batch = importance_scale_batch.clamp(max=10)

                contribs_buf[converged, :, :] = contribs_batch
                if not use_hypothetical:
                    seqs_buf[converged, :, :] = seqs_batch  # type: ignore

                importance_scale_buf[converged, :, :] = importance_scale_batch

                inds_buf[converged] = inds_batch
                global_scale_buf[converged] = global_scale_batch

                coef_inter[converged, :, :] = 0.0
                coef[converged, :, :] = 0.0
                i[converged] = 0
                n_total[converged] = 0
                n_restarts[converged] = 0
                n_reductions[converged] = 0
                gap_min[converged] = float("inf")
                gap_ref[converged] = float("inf")

                step_sizes[converged] = step_size_max

            # Optimization step
            step_count += 1
            check = step_count % check_every == 0
            coef_inter, coef, gap, nll = step_fn(
                cwms_tensor,
                contribs_buf,
                importance_scale_buf,
                seqs_buf,
                coef_inter,
                coef,
                i,
                step_sizes,
                L,
                lambdas_tensor,
                check,
            )
            i += 1
            n_total += 1
            if not check:
                num_load = 0
                continue

            # Assess convergence of each peak being optimized. Reset diverged peaks with lower step size.
            active = inds_buf >= 0

            diverged = ~torch.isfinite(gap)
            if early_restart:
                # The empty fit scores 0.5 per bp; a fit above it is diverging (or overshooting)
                # sum over positions first: no (B, M, P) temporary
                objective = nll + (coef.sum(dim=2) * lambdas_tensor[:, :, 0]).sum(dim=1) / L
                diverged |= objective > 0.5 * (1 + 1e-3)
            diverged &= active
            coef_inter[diverged, :, :] = 0.0
            coef[diverged, :, :] = 0.0
            i[diverged] = 0
            n_restarts += diverged.int()
            step_sizes[diverged, :, :] *= step_adjust

            # Stall safeguard: reduce the step and drop the momentum of regions whose best gap over the
            # last stall_window steps improved by less than stall_improvement on the window before
            finite = active & torch.isfinite(gap)
            gap_min = torch.where(finite, torch.minimum(gap_min, gap), gap_min)
            boundary = active & (n_total % stall_window == 0)
            stalled = (
                boundary
                & (n_total >= stall_start)
                & (gap_min > (1 - stall_improvement) * gap_ref)
            )
            step_sizes[stalled, :, :] *= step_adjust
            coef_inter[stalled, :, :] = coef[stalled, :, :]
            i[stalled] = 0
            n_reductions += stalled.int()
            gap_ref = torch.where(boundary, gap_min, gap_ref)
            gap_min = torch.where(boundary, torch.full_like(gap_min, float("inf")), gap_min)

            timeouts = (n_total > max_steps) & active
            if timeouts.sum().item() > 0:
                timeout_inds = inds_buf[timeouts]
                for ind in timeout_inds:
                    warnings.warn(
                        f"Region {ind} has not converged within max_steps={max_steps} iterations.",
                        RuntimeWarning,
                    )

            fails = (step_sizes < step_size_min).squeeze() & active
            if fails.sum().item() > 0:
                fail_inds = inds_buf[fails]
                for ind in fail_inds:
                    warnings.warn(f"Optimizer failed for region {ind}.", RuntimeWarning)

            converged = ((gap <= convergence_tol) | timeouts | fails) & active
            num_load = converged.sum().item()

            # Extract hits from converged peaks
            if num_load > 0:
                inds_out = inds_buf[converged]
                global_scale_out = global_scale_buf[converged]

                # Compute hit scores
                coef_out = coef[converged, :, :]
                contribs_converged = contribs_buf[converged, :, :]

                # Squared norm of the contributions under each trimmed motif window, computed
                # directly: recovering it from importance_scale as scale ** -2 - eps loses
                # digits in float32 and is wrong wherever the clamp at 10 applies
                importance_sq = F.conv1d(contribs_converged**2, cwm_trim_mask_tensor)
                xcor_scale = importance_sq.sqrt()
                importance_sum_out_dense = F.conv1d(
                    torch.abs(contribs_converged), cwm_trim_mask_tensor
                )
                xcov_out_dense = F.conv1d(contribs_converged, cwms_tensor)
                # xcov_out_dense = F.conv1d(torch.abs(contribs_converged), cwms_tensor)
                xcor_out_dense = xcov_out_dense / xcor_scale

                # Placements the similarity filter removes: many in a region flag a poor fit
                num_placed = (coef_out > 0).sum(dim=(1, 2))
                if post_filter:
                    coef_out = coef_out * (xcor_out_dense >= lambdas_tensor)
                filtered_out = num_placed - (coef_out > 0).sum(dim=(1, 2))

                # Extract hit coordinates using sparse tensor representation
                coef_out = coef_out.to_sparse()

                # Tensor indexing operations for hit extraction
                hit_idxs_out = torch.clone(coef_out.indices())  # Sparse tensor indices
                hit_idxs_out[0, :] = F.embedding(
                    hit_idxs_out[0, :], inds_out[:, None]
                ).squeeze()  # Embedding lookup with complex indexing
                # Map buffer index to peak index

                ind_tuple = torch.unbind(coef_out.indices())
                importance_out = importance_sum_out_dense[ind_tuple]
                importance_sq_out = importance_sq[ind_tuple]
                xcor_out = xcor_out_dense[ind_tuple]

                scores_out_raw = coef_out.values()

                # Store outputs
                gap_out = gap[converged]
                nll_out = nll[converged]
                step_out = i[converged, 0, 0]
                step_sizes_out = step_sizes[converged, 0, 0]

                hit_idxs_lst.append(hit_idxs_out.numpy(force=True).T)
                coefficients_lst.append(scores_out_raw.numpy(force=True))
                similarity_lst.append(xcor_out.numpy(force=True))
                importance_lst.append(importance_out.numpy(force=True))
                importance_sq_lst.append(importance_sq_out.numpy(force=True))

                qc_lsts["nll"].append(nll_out.numpy(force=True))
                qc_lsts["dual_gap"].append(gap_out.numpy(force=True))
                qc_lsts["num_steps"].append(step_out.numpy(force=True))
                qc_lsts["num_steps_total"].append(n_total[converged].numpy(force=True))
                qc_lsts["num_restarts"].append(n_restarts[converged].numpy(force=True))
                qc_lsts["num_step_reductions"].append(n_reductions[converged].numpy(force=True))
                qc_lsts["num_filtered"].append(filtered_out.numpy(force=True))
                qc_lsts["global_scale"].append(global_scale_out.numpy(force=True))
                qc_lsts["step_size"].append(step_sizes_out.numpy(force=True))
                qc_lsts["peak_id"].append(inds_out.numpy(force=True).astype(np.uint32))

                num_complete += num_load
                pbar.update(num_load)

    # Merge outputs into arrays
    hit_idxs = np.concatenate(hit_idxs_lst, axis=0)
    scores_coefficient = np.concatenate(coefficients_lst, axis=0)
    scores_similarity = np.concatenate(similarity_lst, axis=0)
    scores_importance = np.concatenate(importance_lst, axis=0)
    scores_importance_sq = np.concatenate(importance_sq_lst, axis=0)

    hits: Dict[str, ndarray] = {
        "peak_id": hit_idxs[:, 0].astype(np.uint32),
        "motif_id": hit_idxs[:, 1].astype(np.uint32),
        "hit_start": hit_idxs[:, 2],
        "hit_coefficient": scores_coefficient,
        "hit_similarity": scores_similarity,
        "hit_importance": scores_importance,
        "hit_importance_sq": scores_importance_sq,
    }

    qc: Dict[str, ndarray] = {k: np.concatenate(v, axis=0) for k, v in qc_lsts.items()}

    hits_df = pl.DataFrame(hits)
    qc_df = pl.DataFrame(qc)

    return hits_df, qc_df
