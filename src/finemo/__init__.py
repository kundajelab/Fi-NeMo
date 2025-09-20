"""Fi-NeMo: Finding Neural Network Motifs.

A GPU-accelerated motif instance calling tool for identifying transcription factor
binding sites from neural network contribution scores.

Fi-NeMo implements a competitive optimization approach using proximal gradient descent
to identify motif instances by solving a sparse linear reconstruction problem. The
algorithm represents contribution scores as weighted combinations of motif contribution
weight matrices (CWMs) at specific genomic positions.

Key Features
------------
- GPU-accelerated hit calling using PyTorch
- Support for multiple input formats (bigWig, HDF5, TF-MoDISco)
- Competitive motif instance assignment
- Comprehensive evaluation and visualization tools
- Post-processing utilities for hit refinement

Modules
-------
- hitcaller : Core Fi-NeMo algorithm implementation
- data_io : Data input/output utilities
- main : Command-line interface
- evaluation : Performance assessment tools
- visualization : Plotting and report generation
- postprocessing : Hit refinement and analysis

Examples
--------
Basic hit calling workflow:

>>> import finemo
>>> from finemo import data_io, hitcaller
>>>
>>> # Load preprocessed data
>>> sequences, contribs, peaks_df, has_peaks = data_io.load_regions_npz('regions.npz')
>>> cwms, trim_masks = data_io.load_motif_cwms('motifs.h5')
>>>
>>> # Call hits
>>> hits_df, qc_df = hitcaller.fit_contribs(
...     cwms=cwms,
...     contribs=contribs,
...     sequences=sequences,
...     cwm_trim_mask=trim_masks,
...     use_hypothetical=False,
...     lambdas=np.array([0.7] * len(cwms)),
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

See Also
--------
TF-MoDISco : https://github.com/jmschrei/tfmodisco-lite
BPNet : https://github.com/kundajelab/bpnet-refactor
ChromBPNet: https://github.com/kundajelab/chrombpnet
"""

from . import data_io
from . import hitcaller
from . import evaluation
from . import visualization
from . import postprocessing
from . import main

__all__ = [
    "data_io",
    "hitcaller",
    "evaluation",
    "visualization",
    "postprocessing",
    "main",
]
