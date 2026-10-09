"""Unit tests for the shared private helpers. Pure and fast — no model.

Covers the key constants in ``tcri._state.keys``, the statistics primitives in
``tcri._stats._core`` (stars, HDI), the phenotype distances in
``tcri._compute._distance``, and what ``import tcri`` is allowed to do to the process it is
imported into.
"""
import json
import os

import numpy as np
import pytest

from tcri._state import keys as K
from tcri._compute import _distance as D
from tcri._stats import _core as S


def test_keys_constants():
    assert K.P_CT == "tcri_p_ct"
    assert K.X_LOGITS == "X_tcri_logits"
    assert K.METADATA == "tcri_metadata"
    assert K.X_PROBABILITIES == "X_tcri_probabilities"
    # one name per stored thing: no shadow copies of the clone/phenotype keys for a reader
    # to pick the stale one out of. Only the defensively-popped manager stash name remains.
    assert not hasattr(K, "LEGACY_CLONE_KEY")
    assert not hasattr(K, "LEGACY_PHENOTYPE_KEY")
    assert not hasattr(K, "LEGACY_X_PHENOTYPES")
    assert K.LEGACY_MANAGER == "tcri_manager"


def test_import_tcri_does_not_hijack_global_warning_filters():
    """A library must not silence the application's warnings.

    A blanket ``warnings.filterwarnings('ignore')`` at module scope runs on
    ``import tcri`` and silences EVERY warning in the user's session — including this
    package's own guardrails (the K clamp, the param-store-reuse notice, the
    batch_size warning). Narrow message-specific filters are tolerated; a catch-all
    is not.
    """
    import subprocess
    import sys

    code = (
        "import warnings, io, contextlib; import tcri; "
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stderr(buf): warnings.warn('probe', UserWarning)\n"
        "print('VISIBLE' if 'probe' in buf.getvalue() else 'SILENCED')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "VISIBLE" in out.stdout, (
        "importing tcri silenced a user warning — something re-added a blanket "
        f"filterwarnings('ignore'). stdout={out.stdout!r}"
    )


def test_preprocessing_import_is_light():
    """``import tcri`` must not drag in umap.

    umap pulls pynndescent and numba behind it, so importing it at module scope adds
    that compile cost to every ``import tcri``. It is needed only for the opt-in
    ``to_anndata(compute_umap=True)`` path, which imports it locally.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import tcri, sys; print('umap' in sys.modules)"],
        capture_output=True, text=True,
    )
    assert "False" in out.stdout, f"import tcri eagerly loaded umap: {out.stdout!r}"



def test_stars_thresholds():
    assert S.stars(1e-5) == "****"
    assert S.stars(5e-4) == "***"
    assert S.stars(5e-3) == "**"
    assert S.stars(0.04) == "*"
    assert S.stars(0.2) == "ns"


def test_hdi_hugs_skew():
    """The HDI must be narrower than the equal-tailed interval on a skewed sample.

    Every interval the package reports is an HDI, so this is the property that matters: on a
    skewed sample it must exclude the far tail a percentile interval keeps. The equal-tailed
    bound is computed inline from numpy rather than through an API of its own. The second half
    checks that the two agree on a symmetric sample, where they should.
    """
    x = np.array([0, 0, 0, 0, 0.1, 0.2, 5.0])
    hlo, hhi = S.hdi(x, prob=0.8)
    elo, ehi = np.percentile(x, [10, 90])
    assert (hhi - hlo) <= (ehi - elo)          # HDI no wider than ETI on skew
    assert hhi < ehi                            # and it excludes the far tail

    g = np.random.default_rng(0).normal(size=20000)
    assert np.allclose(S.hdi(g, prob=0.94), np.percentile(g, [3, 97]), atol=0.08)


def test_prob_direction():
    p_gt, p_lt = S.prob_direction([1, 1, -1, 1.0])
    assert abs(p_gt - 0.75) < 1e-9 and abs(p_lt - 0.25) < 1e-9


def test_distance_kernels():
    assert abs(D.kl_divergence([1, 0], [1, 0])) < 1e-9        # KL(p‖p)=0
    assert D.kl_divergence([0.9, 0.1], [0.1, 0.9]) > 0        # asymmetric, positive
    assert D.l1_distance([1, 0], [0, 1]) == pytest.approx(2.0)


def test_distance_dispatch():
    assert D.phenotype_distance("l1")([1, 0], [0, 1]) == pytest.approx(2.0)
    assert D.phenotype_distance("dkl") is D.kl_divergence
    f = lambda p, q: 0.0
    assert D.phenotype_distance(f) is f
    with pytest.raises(ValueError):
        D.phenotype_distance("nope")


def test_importing_tcri_does_not_mutate_global_state():
    """A library configures nothing on the caller's behalf.

    ``import tcri`` must leave SLURM_NTASKS/SLURM_NTASKS_PER_NODE in os.environ and the root
    logger alone. Deleting those variables breaks anything else in the process that sizes work
    from them -- a joblib pool, a subprocess srun, a second Trainer -- and calling
    logging.basicConfig(level=INFO) switches on INFO logging for the entire application.

    Runs in a subprocess: this process has already imported tcri, so a mutation made at import
    time would be invisible here.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import json, logging, os
        os.environ["SLURM_NTASKS"] = "7"
        os.environ["SLURM_NTASKS_PER_NODE"] = "3"
        before = logging.getLogger().level
        import tcri
        print(json.dumps({
            "ntasks": os.environ.get("SLURM_NTASKS"),
            "per_node": os.environ.get("SLURM_NTASKS_PER_NODE"),
            "root_level_changed": logging.getLogger().level != before,
            "root_handlers": len(logging.getLogger().handlers),
        }))
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env={**os.environ, "MPLBACKEND": "Agg"})
    assert out.returncode == 0, out.stderr[-2000:]
    state = json.loads(out.stdout.strip().splitlines()[-1])

    assert state["ntasks"] == "7", "import tcri deleted SLURM_NTASKS from os.environ"
    assert state["per_node"] == "3", "import tcri deleted SLURM_NTASKS_PER_NODE from os.environ"
    assert not state["root_level_changed"], (
        "import tcri reconfigured the ROOT logger; that is the application's call, not a "
        "library's. Attach a NullHandler instead."
    )
