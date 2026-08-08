# Installation

TCRi can be installed from source. We recommend using a virtual environment to avoid dependency conflicts.

## Prerequisites

- Python **3.10 or higher**
- pip (Python package installer)

A CUDA-capable GPU is optional but speeds up model fitting; TCRi runs on CPU otherwise.

## Install from Source

TCRi is not yet on PyPI, so install from the repository:

```bash
# Clone the repository
git clone https://github.com/nceglia/tcri.git
cd tcri

# (Recommended) create a virtual environment
python3 -m venv tvenv
source tvenv/bin/activate  # On Windows, use: tvenv\Scripts\activate

# Install the package
pip install .
```

For a development install with the test dependencies:

```bash
pip install -e ".[test]"
```

## Dependencies

TCRi pulls in the following automatically. The heavy scientific stack (PyTorch, Pyro,
scvi-tools) is what enables model fitting:

- numpy, pandas, scipy, scikit-learn
- anndata, scanpy (single-cell data structures)
- torch, pyro-ppl, scvi-tools (probabilistic modelling backbone)
- umap-learn
- matplotlib, seaborn (plotting)
- tqdm

## Verifying the Installation

To verify that TCRi has been installed correctly, run the following in a Python interpreter:

```python
import tcri

# Print the version
print(tcri.__version__)
```

You should see the version number printed out without any errors.
