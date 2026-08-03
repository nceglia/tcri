"""Frozen definitions of the information-theoretic metrics (the *metrics contract*).

Companion to the **model** contract (:mod:`tcri.model._model_contract`), which freezes
the generative mathematics. This module freezes what the **metrics** compute: the
entropies and mutual information over a clone x phenotype joint.

The two are separate on purpose — they are checked by different means. The model
contract is verified by *tracing* ``model()``/``guide()`` for sites, plates and
distribution families. Metrics are pure functions of a joint table, so they are
verified by *numeric identities* (uniform -> log2(k), independent -> MI 0, and the
entropy/MI decomposition) in
``tests/test_metrics_contract_conformance.py``.

Source of truth: **Supplementary Note 1**, "Entropy" section (eqs 2-4) and the mutual
information it defines (eqs 2-6).

Changing any definition here means changing what the published numbers mean.
**Update this manifest and ``docs/contract/METRICS_CONTRACT.md`` FIRST, then the
code** — never loosen a definition to make a failing conformance test pass.
"""
from __future__ import annotations

__all__ = [
    "LOG_BASE",
    "METRIC_SPECS",
    "IDENTITIES",
    "SANCTIONED_EXTENSIONS",
    "MetricSpec",
]

#: All entropies/MI are reported in **bits**. The note writes an unspecified ``log``;
#: tcri fixes base 2 throughout so entropies read as bits and normalizers are log2(k).
LOG_BASE = 2


class MetricSpec:
    """One frozen metric definition."""

    def __init__(self, name, formula, per, support, normalizer, empty, note_eq):
        self.name = name
        self.formula = formula      # the exact quantity computed
        self.per = per              # what one output value corresponds to
        self.support = support      # how zero/absent mass is handled
        self.normalizer = normalizer
        self.empty = empty          # value when there is no mass
        self.note_eq = note_eq

    def __repr__(self):  # pragma: no cover - debug aid
        return f"MetricSpec({self.name!r}, {self.formula!r})"


METRIC_SPECS = {
    "clonotypic_entropy": MetricSpec(
        name="clonotypic_entropy",
        formula="H[P(c|phi)] = -sum_c P(c|phi) log2 P(c|phi)",
        per="one value per PHENOTYPE (how spread that phenotype is across clones)",
        support=(
            "SUPPORT-ONLY: clones with zero mass in the column are dropped BEFORE "
            "renormalizing. No epsilon clip — fabricating uniform mass on absent "
            "clones would inflate H toward 1."
        ),
        normalizer="log2(#supported clones), or log2(n_clones_ref) when given",
        empty="NaN when the phenotype column has no positive mass",
        note_eq="eq 3",
    ),
    "phenotypic_entropy": MetricSpec(
        name="phenotypic_entropy",
        formula="H[P(phi|c)] = -sum_phi P(phi|c) log2 P(phi|c)",
        per="one value per CLONE (plasticity vs commitment of that clone)",
        support=(
            "All P phenotypes are in the sum; 0*log0 is taken as 0. A clone with zero "
            "total mass yields NaN — it is NOT reindexed to zeros, which would report "
            "a spurious H=1 for a clone that was never observed."
        ),
        normalizer="log2(P), P = number of phenotype categories",
        empty="NaN when the clone row has no positive mass",
        note_eq="eq 4",
    ),
    "mutual_information": MetricSpec(
        name="mutual_information",
        formula="I(c;phi) = sum_{c,phi} P(c,phi) log2( P(c,phi) / (P(c) P(phi)) )",
        per="one value per (covariate) joint table",
        support=(
            "The table is renormalized to a joint; a numerical epsilon (1e-15) guards "
            "log(0). Rows/columns with zero mass contribute zero."
        ),
        normalizer=(
            "normalize_mode='min' (DEFAULT): I / min(H(c), H(phi)) — the coefficient of "
            "constraint. 'average': I / (0.5*(H(c)+H(phi))). 'min' is the default "
            "because the 'average' denominator scales with log2(C) and is therefore NOT "
            "comparable across groups with different clone counts."
        ),
        empty="NaN when the table has no positive mass",
        note_eq="the MI defined over the joint in the note's Entropy section",
    ),
}


#: Identities the conformance test enforces. These are what make a redefinition
#: detectable: any change to the metrics that breaks one of these is a contract change.
IDENTITIES = {
    "entropy_uniform_is_log2_k": (
        "A uniform distribution over k supported outcomes has H = log2(k) bits, and "
        "normalized H = 1.0."
    ),
    "entropy_degenerate_is_zero": (
        "All mass on one outcome gives H = 0 (normalized and unnormalized)."
    ),
    "entropy_zero_mass_is_nan": (
        "A clone/phenotype with no posterior mass yields NaN — never 0 and never a "
        "spurious 1 from reindexing absent entries to zeros."
    ),
    "mi_independent_is_zero": (
        "For an independent joint P(c,phi) = P(c)P(phi), I(c;phi) = 0."
    ),
    "mi_is_symmetric": "I(c;phi) == I(phi;c) (transposing the table is a no-op).",
    "mi_is_nonnegative": "I(c;phi) >= 0 for every joint.",
    "mi_perfect_coupling_is_one": (
        "For a permutation-like joint (each clone in exactly one phenotype and vice "
        "versa), normalized MI with mode='min' is 1.0."
    ),
    "mi_entropy_decomposition": (
        "THE cross-metric identity: I(c;phi) = H(c) - sum_phi P(phi) * H[P(c|phi)], "
        "where H[P(c|phi)] is the UNNORMALIZED clonotypic entropy. This ties the two "
        "metric families together — it is the identity that proved the note's eqs 3-4 "
        "are mistranscribed (weighting by the marginal instead of the conditional "
        "makes this yield a NEGATIVE mutual information)."
    ),
}


#: Deliberate additions beyond the note. Not deviations from its mathematics — the
#: note simply does not specify them.
SANCTIONED_EXTENSIONS = {
    "bits_log2": (
        "The note writes an unspecified `log`; tcri fixes base 2 so all entropies are "
        "in bits and normalizers are log2(k)."
    ),
    "normalization": (
        "`normalized=True` divides by the maximum-entropy value so results land in "
        "[0,1] and compare across groups of different size. The note defines only the "
        "raw entropies for eqs 3-4."
    ),
    "normalize_mode_default": (
        "DEVIATION FROM eq 6. The note's eq 6 defines "
        "NMI = I / ((1/2)(H(c) + H(phi))) -- the MEAN denominator, exposed here as "
        "normalize_mode='average'. tcri's DEFAULT is 'min' (I / min(H(c), H(phi))), "
        "the coefficient of constraint, because the mean denominator scales with "
        "log2(C) and is therefore NOT comparable across groups with different clone "
        "counts -- the blocking issue for any per-group or per-patient comparison. "
        "The two differ materially (0.293 vs 0.239 on the contract's test joint), so "
        "anything reproducing the note's benchmark MUST pass "
        "normalize_mode='average' explicitly. Pinned by "
        "test_eq6_nmi_is_the_average_denominator, which asserts BOTH that 'average' "
        "reproduces eq 6 and that the default does not -- so the divergence can never "
        "become silent."
    ),
    "n_clones_ref": (
        "clonotypic_entropy accepts `n_clones_ref` to FIX the normalizer across groups; "
        "without it each group is normalized by its own supported-clone count, which is "
        "not comparable between groups."
    ),
    "posterior_summaries": (
        "`n_samples>0` returns mean/sd/HDI of the metric over posterior draws instead of "
        "a single plug-in value. The plug-in entropy is >= the posterior mean (Jensen), "
        "so the two are reported as distinct quantities."
    ),
}
