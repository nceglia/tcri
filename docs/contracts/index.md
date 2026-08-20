# Governance

TCRi is governed by four **frozen, machine-checked contracts**. Each pins one facet of the
package — the public interface, the generative mathematics, the metric definitions, and the
training plan — so that a change to any of them is deliberate and reviewed rather than
accidental.

## Why contracts

A published number — a mutual information in bits, a phenotype call — only means something if
the definition behind it is stable. These contracts make each definition explicit and enforce
it with a test, so that:

- the model the package *claims* to implement is the model it *does* implement;
- a metric cannot be silently redefined, which would change what every past figure means;
- the training plan has recorded, behavioural bounds rather than folklore defaults.

**A failing conformance test means _stop and decide_** — is this an intended change, in which
case the contract is updated first and deliberately, or a regression, in which case the code
is fixed? It never means "loosen the manifest until it passes."

## The four contracts

| Contract | Freezes |
|----------|---------|
| **API** | the public interface — which namespaces exist and the signature of every function in them |
| **Model** | the generative mathematics — each stochastic site, its distribution family and plate, and the objective |
| **Metrics** | what the metrics compute — the entropies and mutual information over the clone × phenotype joint |
| **Training** | how the model is fit — the schedule, the stopping rule, and what each knob actually does |

The model contract is verified by *tracing* the live model and guide and asserting every site,
scale and index map behaviourally — a source-text or scalar check is not enough. The metrics
contract instead pins **numeric identities**, because metrics are pure functions of a joint
table. The keystone is

$$I(c;\phi) = H(c) - \sum_\phi P(\phi)\,H[P(c\mid\phi)],$$

which ties the entropy and MI families together so that neither can be redefined alone.

## What this means for you

If you are **using** TCRi, the practical consequence is that the definitions on the
[concepts page](../concepts/index.md) are the definitions, and they will not change under you
without a version bump and a note.

If you are **contributing**, the contracts live in `governance/` alongside the manifests they
freeze, and the conformance tests are in `tests/`. Only the maintainers may change a contract,
a conformance test, or a source document; this is enforced by code owners on the repository.
The contributor-facing detail — the equation-by-equation code map, the recorded departures and
their rationales, and the open questions — lives there rather than here, because it is a record
for people changing the package, not for people using it.
