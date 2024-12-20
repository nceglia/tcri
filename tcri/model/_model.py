import pyro
import torch
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pyro.distributions as dist
from torch.distributions import constraints
import matplotlib.pyplot as plt

class AnnDataTCRDataset(Dataset):
    def __init__(self, adata, gene_list, tcr_key):
        self.adata = adata
        self.gene_list = gene_list
        self.tcr_key = tcr_key

        self.gene_indices = [np.where(adata.var_names == gene)[0][0] for gene in gene_list]
        self.gene_expression = self.adata.X[:, self.gene_indices].toarray() if hasattr(self.adata.X, "toarray") else self.adata.X[:, self.gene_indices]
        self.tcrs = self.adata.obs[self.tcr_key].astype('category').cat.codes.values
        self.unique_tcrs = self.adata.obs[self.tcr_key].astype('category').cat.categories

    def __len__(self):
        return self.gene_expression.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.gene_expression[idx], dtype=torch.float32)  # Gene expression
        z = torch.tensor(self.tcrs[idx], dtype=torch.long)  # TCR
        return x, z

    def get_tcr_mapping(self):
        return dict(enumerate(self.unique_tcrs))

def model(genes, tcrs, K, D):
    """
    Args:
        genes: Batch of gene expression data (batch_size, D)
        tcrs: Batch of TCR indices (batch_size)
        K: Number of unique TCRs
        D: Number of genes
    """
    probs = pyro.sample("probs", dist.Dirichlet(torch.ones(K, device=genes.device)))  # Shape: (K,)

    with pyro.plate("tcr_plate", K, dim=-2):  # Plate over K (TCRs)
        with pyro.plate("gene_plate", D, dim=-1):  # Plate over D (genes)
            mu = pyro.sample(
                "mu", 
                dist.LogNormal(
                    torch.zeros((K, D), device=genes.device),  # Shape: (K, D)
                    torch.ones((K, D), device=genes.device) * 1.0  # Shape: (K, D)
                )
            )  # Shape: (K, D)
            theta = pyro.sample(
                "theta", 
                dist.Gamma(
                    torch.ones((K, D), device=genes.device) * 2.0,  # Shape: (K, D)
                    torch.ones((K, D), device=genes.device) * 0.5  # Shape: (K, D)
                )
            )  # Shape: (K, D)
            gate = pyro.sample(
                "gate", 
                dist.Beta(
                    torch.ones((K, D), device=genes.device) * 0.8, 
                    torch.ones((K, D), device=genes.device) * 0.8
                )
            )  # Shape: (K, D)

    with pyro.plate("data", size=genes.shape[0], dim=-2):  # Batch over data (batch_size)
        z = pyro.sample("z", dist.Categorical(probs), obs=tcrs)  # Shape: (batch_size,)

        mu_z = mu[z]  # Shape: (batch_size, D)
        theta_z = theta[z]  # Shape: (batch_size, D)
        gate_z = gate[z]  # Shape: (batch_size, D)

        probs = torch.sigmoid(mu_z / (mu_z + theta_z))  # Ensure this is within [0, 1]
        
        zinb = dist.ZeroInflatedNegativeBinomial(
            total_count=theta_z,
            probs=probs, 
            gate=gate_z  
        )
        pyro.sample("genes", zinb.to_event(1), obs=genes)  

def guide(genes, tcrs, K, D):
    probs_q = pyro.param("probs_q", torch.ones(K, device=genes.device), constraint=constraints.simplex)
    pyro.sample("probs", dist.Dirichlet(probs_q))  # Shape: (K,)

    with pyro.plate("tcr_plate", K, dim=-2):  # Plate over K (TCRs)
        with pyro.plate("gene_plate", D, dim=-1):  # Plate over D (genes)
            mu_q = pyro.param("mu_q", torch.zeros((K, D), device=genes.device))  # (K, D)
            sigma_q = pyro.param("sigma_q", torch.ones((K, D), device=genes.device), constraint=constraints.positive)  # Shape: (K, D)
            pyro.sample("mu", dist.LogNormal(mu_q, sigma_q))  #(K, D)

            theta_q = pyro.param("theta_q", torch.ones((K, D), device=genes.device), constraint=constraints.positive)  # Shape: (K, D)
            pyro.sample("theta", dist.Gamma(theta_q, 1.0))  # (K, D)

            gate_alpha_q = pyro.param("gate_alpha_q", torch.ones((K, D), device=genes.device), constraint=constraints.positive)  # Shape: (K, D)
            gate_beta_q = pyro.param("gate_beta_q", torch.ones((K, D), device=genes.device), constraint=constraints.positive)  # Shape: (K, D)
            pyro.sample("gate", dist.Beta(gate_alpha_q, gate_beta_q))  # (K, D)
    
    with pyro.plate("data", size=genes.shape[0], dim=-2):  
        z_logits = pyro.param("z_logits", torch.randn((genes.shape[0], K), device=genes.device))  # Shape: (batch_size, K)
        pyro.sample("z", dist.Categorical(logits=z_logits), infer={"is_auxiliary": True})  # Shape: (batch_size,)


class JointProbabilityDistribution(object):
    def __init__(self, adata, gene_list, batch_size=500, lr=0.01):
        self.adata = adata
        self.losses = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.dataset = AnnDataTCRDataset(adata, gene_list, adata.uns["tcri_clone_key"])
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.K = len(self.dataset.get_tcr_mapping())  # Number of TCRs
        self.D = len(gene_list)  # Number of genes

        pyro.clear_param_store()
        self.svi = SVI(
            model=lambda genes, tcrs, K, D: model(genes, tcrs, K, D), 
            guide=lambda genes, tcrs, K, D: guide(genes, tcrs, K, D), 
            optim=pyro.optim.Adam({"lr": lr}), 
            loss=Trace_ELBO()
        )

    def train(self, num_epochs, print_every=5):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (gene_batch, tcr_batch) in enumerate(self.data_loader):
                gene_batch = gene_batch.to(self.device)
                tcr_batch = tcr_batch.to(self.device)
                loss = self.svi.step(gene_batch, tcr_batch, K=self.K, D=self.D)
                epoch_loss += loss
                self.losses.append(loss)
            if epoch % print_every == 0:
                print(f"Epoch {epoch} Loss: {epoch_loss / len(self.data_loader):.4f}")
    
    def plot_loss(self):
        """ Plot training loss """
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label="ELBO Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()

