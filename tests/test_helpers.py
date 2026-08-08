"""Unit tests for the shared helper modules introduced in PR1
(`_keys`, `_console`, `_stats`, `_distance`). These are pure and fast — no model."""
import json
import os

import numpy as np
import pytest

from tcri import _console, _distance as D, _keys as K, _stats as S


def test_keys_constants():
    assert K.P_CT == "tcri_p_ct"
    assert K.X_LOGITS == "X_tcri_logits"
    assert K.METADATA == "tcri_metadata"
    assert K.X_PROBABILITIES == "X_tcri_probabilities"
    # the legacy shadow keys are GONE (Phase 4 removal completed); only the
    # defensively-popped manager stash name remains
    assert not hasattr(K, "LEGACY_CLONE_KEY")
    assert not hasattr(K, "LEGACY_PHENOTYPE_KEY")
    assert not hasattr(K, "LEGACY_X_PHENOTYPES")
    assert K.LEGACY_MANAGER == "tcri_manager"


def test_import_tcri_does_not_hijack_global_warning_filters():
    """A library must not silence the application's warnings.

    ``_preprocessing`` used to call a blanket ``warnings.filterwarnings('ignore')``
    at module scope, which runs on ``import tcri`` and silenced EVERY warning in the
    user's session — including this package's own guardrails (the K clamp, the
    param-store-reuse notice, the batch_size warning). Narrow message-specific
    filters are tolerated; a catch-all is not.
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
    """``import tcri`` must not drag in umap (~2.9 s via pynndescent/numba).

    umap is only needed for the opt-in ``to_anndata(compute_umap=True)`` path, which
    imports it locally. An eager module-level import cost every user ~2.9 s.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import tcri, sys; print('umap' in sys.modules)"],
        capture_output=True, text=True,
    )
    assert "False" in out.stdout, f"import tcri eagerly loaded umap: {out.stdout!r}"


def test_console_aliases_and_callables():
    assert _console.MAG == _console.MAGENT
    assert _console.GRN == _console.GREEN
    assert _console.CYN == _console.CYAN
    for fn in (_console._ok, _console._info, _console._warn, _console._fin):
        callable(fn)
    _console._ok("x", quiet=True)  # quiet path prints nothing, must not raise


def test_stars_thresholds():
    assert S.stars(1e-5) == "****"
    assert S.stars(5e-4) == "***"
    assert S.stars(5e-3) == "**"
    assert S.stars(0.04) == "*"
    assert S.stars(0.2) == "ns"


def test_hdi_hugs_skew_and_matches_eti_on_symmetric():
    x = np.array([0, 0, 0, 0, 0.1, 0.2, 5.0])
    hlo, hhi = S.hdi(x, prob=0.8)
    elo, ehi = S.eti(x, prob=0.8)
    assert (hhi - hlo) <= (ehi - elo)          # HDI no wider than ETI on skew
    assert hhi < ehi                            # and it excludes the far tail
    g = np.random.default_rng(0).normal(size=20000)
    assert np.allclose(S.hdi(g, prob=0.94), S.eti(g, prob=0.94), atol=0.08)


def test_prob_direction():
    p_gt, p_lt = S.prob_direction([1, 1, -1, 1.0])
    assert abs(p_gt - 0.75) < 1e-9 and abs(p_lt - 0.25) < 1e-9


def test_auc_helpers_on_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9]); labels = np.array([0, 0, 1, 1])
    auc, p_perm, perm, mode = S.auc_and_label_permutation(scores, labels)
    assert auc == 1.0 and mode == "exact" and 0.0 <= p_perm <= 1.0
    lo, hi = S.bootstrap_auc(scores, labels)
    assert 0.0 <= lo <= hi <= 1.0


def test_auc_permutation_matches_sklearn_exactly_with_ties():
    """The Mann–Whitney rank-sum identity must reproduce ``roc_auc_score`` exactly.

    Guards the O(n log n) -> O(n_pos) optimization of the permutation loop. Ties are
    the failure mode that matters: the identity is only equivalent under *midranks*,
    so scores are rounded here to force repeated values.
    """
    import itertools

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(4, 11))
        scores = np.round(rng.normal(size=n), int(rng.integers(0, 2)))  # forces ties
        labels = rng.integers(0, 2, n)
        if labels.sum() in (0, n):
            continue
        auc, _, perm, mode = S.auc_and_label_permutation(scores, labels)
        assert mode == "exact"
        y = (labels == 1).astype(int)
        assert auc == pytest.approx(roc_auc_score(y, scores), abs=1e-12)
        ref = sorted(
            roc_auc_score(np.isin(np.arange(n), idx).astype(int), scores)
            for idx in itertools.combinations(range(n), int(y.sum()))
        )
        np.testing.assert_allclose(sorted(perm), ref, atol=1e-12)


def test_auc_permutation_degenerate_single_class():
    """A single-class label vector has no defined AUROC — report, don't crash."""
    auc, p, perm, mode = S.auc_and_label_permutation(
        np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1])
    )
    assert mode == "degenerate" and np.isnan(p) and perm.size == 0


def test_distance_kernels():
    assert abs(D.kl_divergence([1, 0], [1, 0])) < 1e-9        # KL(p‖p)=0
    assert D.kl_divergence([0.9, 0.1], [0.1, 0.9]) > 0        # asymmetric, positive
    assert D.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-6)  # ~1 bit disjoint
    assert D.jensen_shannon([0.5, 0.5], [0.5, 0.5]) < 1e-9    # symmetric, self=0
    assert D.l1_distance([1, 0], [0, 1]) == pytest.approx(2.0)


def test_distance_dispatch():
    assert D.phenotype_distance("l1")([1, 0], [0, 1]) == pytest.approx(2.0)
    assert D.phenotype_distance("dkl") is D.kl_divergence
    f = lambda p, q: 0.0
    assert D.phenotype_distance(f) is f
    with pytest.raises(ValueError):
        D.phenotype_distance("nope")


def test_importing_tcri_does_not_mutate_global_state():
    """NEW-5: a library configures nothing on the caller's behalf.

    ``import tcri`` used to delete SLURM_NTASKS/SLURM_NTASKS_PER_NODE from os.environ and call
    logging.basicConfig(level=INFO) at module scope. The first breaks anything else in the
    process that sizes work from those variables -- a joblib pool, a subprocess srun, a second
    Trainer -- and the second switches on INFO logging for the entire application.

    Runs in a subprocess: this process has already imported tcri, so the mutation would be
    invisible here.
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
