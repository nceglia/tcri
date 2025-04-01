# Model API

## TCRIModel

The `TCRIModel` class implements a hierarchical Bayesian model for analyzing TCR and gene expression data.

```python
class TCRIModel:
    """
    TCRi Model for joint analysis of gene expression and TCR data.
    
    This model implements a hierarchical Bayesian framework that learns
    a joint representation of gene expression and TCR sequences.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression and TCR information
    n_latent : int, default=10
        Dimension of the latent space
    n_hidden : int, default=128
        Number of hidden units in the neural networks
    global_scale : float, default=10.0
        Scale parameter for the global prior
    local_scale : float, default=5.0
        Scale parameter for the local prior
    sharp_temperature : float, default=1.0
        Temperature parameter for sharpening the distributions
    sharpness_penalty_scale : float, default=0.0
        Scale for the sharpness penalty
    use_enumeration : bool, default=False
        Whether to use enumeration for discrete variables
    device : str, optional
        Device to use for computation ("cpu" or "cuda")
    """
    
    def train(self, max_epochs=50, batch_size=128, lr=1e-3, 
    margin_scale=0.0, margin_value=2.0, adaptive_margin=False, 
    reconstruction_loss_scale=1e-2, n_steps_kl_warmup=1000):
        """
        Train the model.
        
        Parameters
        ----------
        max_epochs : int, default=50
            Maximum number of epochs to train for
        batch_size : int, default=128
            Batch size for training
        lr : float, default=1e-3
            Learning rate
        margin_scale : float, default=0.0
            Scale for the margin loss
        margin_value : float, default=2.0
            Value for the margin
        adaptive_margin : bool, default=False
            Whether to use adaptive margin
        reconstruction_loss_scale : float, default=1e-2
            Scale for the reconstruction loss
        n_steps_kl_warmup : int, default=1000
            Number of steps for KL warmup
        """
        pass
    
    def get_latent_representation(self, adata=None, batch_size=256):
        """
        Get the latent representation for the data.
        
        Parameters
        ----------
        adata : AnnData, optional
            AnnData object to get latent representation for.
            If None, uses the training data.
        batch_size : int, default=256
            Batch size for inference
            
        Returns
        -------
        ndarray
            Latent representation of shape (n_cells, n_latent)
        """
        pass
    
    def get_phenotype_probabilities(self, adata=None, batch_size=256):
        """
        Get phenotype probabilities for the data.
        
        Parameters
        ----------
        adata : AnnData, optional
            AnnData object to get probabilities for.
            If None, uses the training data.
        batch_size : int, default=256
            Batch size for inference
            
        Returns
        -------
        ndarray
            Phenotype probabilities of shape (n_cells, n_phenotypes)
        """
        pass
    
    def save(self, path):
        """
        Save the model to a file.
        
        Parameters
        ----------
        path : str
            Path to save the model to
        """
        pass
    
    @classmethod
    def load(cls, path, adata=None):
        """
        Load a model from a file.
        
        Parameters
        ----------
        path : str
            Path to load the model from
        adata : AnnData, optional
            AnnData object to use with the model
            
        Returns
        -------
        TCRIModel
            Loaded model
        """
        pass
```

## Usage Example

```python
import tcri
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize model
model = tcri.TCRIModel(
    adata,
    n_latent=10,
    n_hidden=128,
    global_scale=10.0,
    local_scale=5.0
)

# Train model
model.train(
    max_epochs=50,
    batch_size=128,
    lr=1e-3,
    reconstruction_loss_scale=1e-2
)

# Get latent representations
latent_z = model.get_latent_representation(adata)

# Get phenotype probabilities
probs = model.get_phenotype_probabilities(adata)

# Save model
model.save("your_model.pkl")

# Load model
loaded_model = tcri.TCRIModel.load("your_model.pkl", adata)
```
