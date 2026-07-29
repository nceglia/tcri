from scipy.stats import entropy
from .. import _keys as K
import numpy as np
import tqdm
import pandas as pd
import collections
import warnings
import torch
import torch.nn.functional as F
import datetime
import pyro.distributions as dist
from pyro.distributions import Dirichlet
import pyro
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional
import umap
import numpy as np, pandas as pd, torch, umap
from tqdm.auto import tqdm
from scvi import REGISTRY_KEYS

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
import pandas as pd
from   scipy.special import softmax
from   torch.distributions import Dirichlet
import torch
import warnings

warnings.filterwarnings('ignore')

# ------------ simple ANSI helpers ------------ #
RESET  = "\x1b[0m"
BOLD   = "\x1b[1m"
DIM    = "\x1b[2m"
GREEN  = "\x1b[32m"
CYAN   = "\x1b[36m"
MAGENT = "\x1b[35m"
# ╭─ colour / pretty-print helpers ─────────────────────────────────────────╮
RESET  = "\x1b[0m";  BOLD  = "\x1b[1m";  DIM  = "\x1b[2m"
GRN = "\x1b[32m";  CYN = "\x1b[36m";  MAG = "\x1b[35m";  YLW = "\x1b[33m"; RED = "\x1b[31m"

from .._console import _ok, _info, _warn, _fin

def group_singletons(adata,clonotype_key="trb",groupby="patient", target_col="trb_unique", min_clone_size=10):
    adata.obs["trb_candidate"] = adata.obs[clonotype_key].astype(str) + "_" + adata.obs[groupby].astype(str)
    clone_counts = adata.obs["trb_candidate"].value_counts()
    def collapse_singleton(row):
        candidate = row["trb_candidate"]
        if clone_counts[candidate] < min_clone_size:
            return f"Singleton_{row[groupby]}"
        else:
            return candidate
    adata.obs[target_col] = adata.obs.apply(collapse_singleton, axis=1)



# ------------ helper to extract logits -------- #

# ------------ main routine -------------------- #









def clone_size(adata, key_added=K.CLONE_SIZE, return_counts=False):
    # Canonical source is uns[METADATA]['clone_col'] (written by to_anndata). This
    # used to read the legacy uns['tcri_clone_key'] shadow key — the last reader of
    # it, which is why the shim outlived Phase 4.
    meta = adata.uns.get(K.METADATA)
    if not meta or K.CLONE_COL not in meta:
        raise KeyError(
            f"adata.uns[{K.METADATA!r}][{K.CLONE_COL!r}] is missing — run "
            "model.to_anndata(adata) first (or load a session) so the clonotype "
            "column is registered."
        )
    tcr_key = meta[K.CLONE_COL]
    res = np.unique(adata.obs[tcr_key].tolist(), return_counts=True)
    clone_sizes = dict(zip(res[0],res[1]))
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes


