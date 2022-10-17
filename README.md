# **TCR**i
Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing

https://www.biorxiv.org/content/10.1101/2022.10.01.510457v1


Install:

```
python3 -m venv tvenv
source tvenv/bin/activate
python3 setup.py install
```


Example code found in /examples.

### Basics

#### Compute joint distribution with TCRi on anndata processed with GeneVectgor.
```
import tcri

sample_column="sample"
condition_column="source"
phenotype_column="genevector"

tcri.pp.joint_distribution(adata,
                           sample_column=sample_column, 
                           condition_column=condition_column, 
                           phenotype_column=phenotype_column)
```
                
#### Each metric has an associated plotting function
1. Clonotypic Entropy
    ```
    tcri.metrics.clonotypic_entropy(adata)
    tcri.pl.clonotypic_entropy(adata)
    ```
2. Phenotypic Entropy
    ```
    tcri.metrics.phenotypic_entropy(adata)
    tcri.pl.phenotypic_entropy(adata)
    ```
3. Phenotypic Flux
    ```
    tcri.metrics.phenotypic_flux(adata,from_this="TP1",to_this="TP2")
    tcri.pl.phenotypic_flux(adata,from_this="TP1",to_this="TP2")
    ```


