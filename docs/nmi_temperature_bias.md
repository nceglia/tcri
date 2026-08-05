# A coupling-strength-dependent bias in the NMI estimate

**Status:** open. Mechanism identified, magnitude bounded, fix not yet decided.
**Scope:** `tcri.tl.mutual_information` at `n_samples > 0`. Evidence is synthetic.

---

## 1. What the testing set out to do

The package reports normalized mutual information between clonotype and phenotype. That
number appears in published figures, so it needs a calibration: given data where the true
clone→phenotype coupling is known exactly, does the estimator recover it?

The test bed generates cells from the *same fitted parameters* behind the published
benchmark — 47 clonotypes, 984 genes, K=10 phenotypes — so the estimate is comparable to
the published numbers rather than to a toy problem. Coupling strength is swept by
sharpening or flattening the fitted ω, and the generator reproduces the published
ground-truth anchors exactly (0.520 / 0.316 / 0.182).

Each grid cell carries three reference points:

- **truth** — the population NMI, computed in closed form from ω.
- **label oracle** — the plug-in NMI over the realized cells' true labels. What a method
  with perfect label knowledge would report on exactly this sample.
- **GMM** — a clustering baseline, as an independent control.

## 2. What the test revealed

TCRi's estimate is biased **upward**, and the size of the bias depends on how strong the
true coupling is. At the flattest setting it reads 0.267 where the sample's own true
labels give 0.191 and the truth is 0.182.

The direction never flips. The estimator over-reads at every coupling strength; it simply
over-reads more when there is less real structure to find.

Two properties made this worth pursuing rather than dismissing as sampling noise:

- The **label oracle already absorbs finite-sample bias**. At N=5000 the oracle sits
  0.010 above truth, so an estimate 0.076 above the oracle is claiming structure the
  realized sample does not contain.
- **GMM is flat across the sweep** (error 0.005–0.033, no trend). The dependence is
  specific to this estimator, not a property of the metric or the test bed.

## 3. How it was confirmed

**It is not one fixture.** Three independently fitted parameter sets, each run at its own
K, reproduce the pattern. Error at N=5000, 1000 epochs:

| coupling | K8 | K10 | K12 |
|---|---|---|---|
| sharp (T=0.1) | 0.013 | 0.015 | 0.017 |
| flat (T=1.0) | 0.035 | 0.058 | 0.061 |
| ratio | 2.7× | 3.8× | 3.7× |

**It is not under-training.** An epoch ladder run to plateau at the flat setting, N=5000:

| epochs | estimate | error |
|---|---|---|
| 60 | 0.2673 | 0.086 |
| 1000 | 0.2397 | 0.058 |
| 2000 | 0.2311 | 0.050 |
| 4000 | 0.2270 | 0.045 |
| 8000 | 0.2272 | 0.046 |

Between 4000 and 8000 epochs the estimate moves by 0.0002. There is a real floor that
training does not remove. The early rows fall steeply, which is why a short run looks like
a convergence problem — the default 60-epoch budget lands in exactly that misleading zone.

**The mechanism is two opposing distortions, not one.** The reported value passes through
two steps that each bias it, in opposite directions:

- The per-cell blend applies a square root to each clone's phenotype row, flattening it.
  This pushes the estimate **down**.
- The posterior is summarized as the mean of the per-draw NMI, `E[NMI(J)]`, rather than
  the NMI of the mean joint, `NMI(E[J])`. NMI is nonlinear in the joint, and each Dirichlet
  draw is sharper than the posterior mean it came from, so this pushes the estimate **up**.

The reported number is the residual of the two. That is why the sharp regime looked
accurate: there the two happened to nearly cancel.

**The nonlinearity term is itself coupling-dependent**, measured at 4000 epochs: +0.017 at
sharp coupling versus +0.099 at flat coupling. The Dirichlet concentration is
`local_scale × p_ct`, so its total is fixed at `local_scale` regardless of the data. With
`local_scale = 3` over 10 phenotypes the per-entry concentration is well below 1, which
makes draws corner-seeking — and flatter rows produce more dispersed draws, hence a larger
gap. This is the first independent support for the proposed mechanism.

## 4. Magnitude and consequence

**The floor is roughly 0.02 to 0.05 NMI**, depending on coupling strength — about 0.045 at
flat coupling and 0.02 at sharp, once trained to plateau.

**The coupling dependence is about 2×**, not the ~12× the default configuration suggests.
Most of that apparent 12× was the short training budget, and it disappears with a longer
run.

Three consequences follow:

- **Cross-regime comparisons are not supportable as printed.** A figure claiming a method
  ranking that changes with coupling strength is reading a bias that changes with coupling
  strength. The ordering of methods within a single regime is less affected.
- **The accuracy at sharp coupling was partly an artifact of the default budget.** Trained
  longer, the error at sharp coupling *grows* (0.007 → 0.022), because the cancellation
  that made it look accurate degrades.
- **The obvious fix does not work alone.** Switching to `NMI(E[J])` removes a term larger
  than the error itself, converting a +0.045 over-read into a −0.054 under-read. Correcting
  one distortion without the other makes accuracy worse.

Separately, the benchmark's smallest cells are not interpretable at all: at N=250 with flat
coupling, a label-permutation null — an estimator with no information — scores 0.214
against a truth of 0.182. Any method evaluated there is being scored above its own noise
floor.

## 5. Next steps

1. **Sweep `local_scale`.** It is the single knob predicted to move the nonlinearity term
   and the floor together. If raising it collapses both, the mechanism is confirmed and the
   fix is a calibration rather than a redesign.
2. **Decide the posterior summary convention.** `E[NMI(J)]` and `NMI(E[J])` are different
   quantities and the package should state which it reports and why. This belongs in the
   metrics contract before it goes in the code, and it affects every metric that accepts
   `n_samples > 0`, not only mutual information.
3. **Resolve the concentration question against Supplementary Note 1.** Total posterior
   concentration does not scale with a clone's cell count, so the reported uncertainty is
   set by a prior rather than informed by data. This is β on eq 2 — a model-contract
   decision, not a code change.
4. **Report the noise floor per benchmark cell.** A permutation null costs almost nothing
   and would have flagged the small-N cells automatically.
5. **Check whether any of this is visible on real data.** Everything above is synthetic.
   Two diagnostics work without ground truth and can run on a real dataset: the value the
   pipeline reports on a table with the clone structure removed, and whether posterior
   width narrows as clones get more cells.

## What is not yet known

- Whether the plateau holds beyond 8000 epochs (16000-epoch runs pending).
- Whether the effect appears under the package's default `normalize_mode="min"`; all of
  the above uses `"average"`.
- Whether the estimator behaves this way on real repertoire data.
