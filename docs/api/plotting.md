# Plotting API

The `tcri.plotting` module (imported as `tcri.pl`) provides functions for visualizing the results of TCRi analyses.

## polar_plot

```python
def polar_plot(adata, phenotypes=None, statistic="distribution", method="joint_distribution", splitby=None, color_dict=None, temperature=1.0):
    """
    Create a polar plot of phenotype distributions or statistics.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    phenotypes : list, optional
        List of phenotype names to include
    statistic : str, default="distribution"
        Statistic to plot, one of "distribution", "entropy", "mi"
    method : str, default="joint_distribution"
        Method to compute distributions
    splitby : str, optional
        Variable to split the plot by
    color_dict : dict, optional
        Dictionary mapping phenotypes to colors
    temperature : float, default=1.0
        Temperature parameter for distributions
        
    Returns
    -------
    matplotlib.figure.Figure
        The polar plot figure
    """
    pass
```

## probability_ternary

```python
def probability_ternary(adata, phenotype_names, splitby=None, conditions=None, top_n=None):
    """
    Create a ternary plot of phenotype probabilities.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    phenotype_names : list
        List of exactly 3 phenotype names to plot
    splitby : str, optional
        Variable to split the plot by
    conditions : list, optional
        List of condition values to include
    top_n : int, optional
        Number of top clones to highlight
        
    Returns
    -------
    matplotlib.figure.Figure
        The ternary plot figure
    """
    pass
```

## mutual_information

```python
def mutual_information(adata, splitby=None, temperature=1.0, n_samples=0, 
                     normalized=True, palette=None, save=None, 
                     legend_fontsize=6, bbox_to_anchor=(1.15,1.), 
                     figsize=(8,4), rotation=90, weighted=True, 
                     return_plot=True):
    """
    Plot mutual information between phenotypes and TCR clonotypes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    splitby : str, optional
        Variable to split the plot by
    temperature : float, default=1.0
        Temperature parameter for distributions
    n_samples : int, default=0
        Number of samples to draw
    normalized : bool, default=True
        Whether to normalize the mutual information
    palette : dict, optional
        Color palette for the plot
    save : str, optional
        Path to save the figure
    legend_fontsize : int, default=6
        Font size for the legend
    bbox_to_anchor : tuple, default=(1.15,1.)
        Position for the legend
    figsize : tuple, default=(8,4)
        Figure size
    rotation : int, default=90
        Rotation for x-axis labels
    weighted : bool, default=True
        Whether to weight by clone size
    return_plot : bool, default=True
        Whether to return the plot
        
    Returns
    -------
    matplotlib.figure.Figure, optional
        The mutual information plot
    """
    pass
```

## clonality

```python
def clonality(adata, groupby=None, splitby=None, s=10, order=None, 
             figsize=(12,5), palette=None):
    """
    Plot clonality metrics.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with TCR information
    groupby : str, optional
        Variable to group by
    splitby : str, optional
        Variable to split by
    s : int, default=10
        Point size
    order : list, optional
        Order of categories
    figsize : tuple, default=(12,5)
        Figure size
    palette : dict, optional
        Color palette
        
    Returns
    -------
    matplotlib.figure.Figure
        The clonality plot
    """
    pass
```

## Usage Examples

```python
import tcri
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize and train model
model = tcri.TCRIModel(adata)
model.train()
tcri.pp.register_model(adata, model)

# Create plots
tcri.pl.polar_plot(adata, statistic="distribution")

tcri.pl.probability_ternary(
    adata,
    ["Phenotype1", "Phenotype2", "Phenotype3"],
    splitby="condition"
)

tcri.pl.mutual_information(
    adata,
    splitby="timepoint",
    figsize=(10,6)
)

tcri.pl.clonality(
    adata,
    groupby="condition",
    splitby="timepoint"
)
```
