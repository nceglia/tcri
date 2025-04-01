# Installation

TCRi can be installed from source. We recommend using a virtual environment to avoid dependency conflicts.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Install from Source

```bash
# Clone the repository
git clone https://github.com/nceglia/tcri.git
cd tcri

# Create a virtual environment
python3 -m venv tvenv
source tvenv/bin/activate  # On Windows, use: tvenv\Scripts\activate

# Install the package
python3 setup.py install
```

## Dependencies

TCRi requires the following packages, which will be installed automatically when you install TCRi:

- scipy
- numpy
- scanpy
- pandas
- matplotlib
- seaborn
- tqdm
- scikit-learn

## Verifying the Installation

To verify that TCRi has been installed correctly, run the following in a Python interpreter:

```python
import tcri

# Print the version
print(tcri.__version__)
```

You should see the version number printed out without any errors.
