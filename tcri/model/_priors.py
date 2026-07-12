"""Prior distributions for the TCRI generative model.

`VampPrior` is a learnable mixture-of-encoders prior over the latent space;
`MixtureDirichlet` is the per-clonotype mixture-of-Dirichlets used as the
clonotype-level phenotype prior.
"""
import torch
import pyro.distributions as dist

__all__ = ["MixtureDirichlet", "VampPrior"]


class VampPrior(torch.nn.Module):
    def __init__(self, pseudo_inputs, encoder):
        """
        Args:
            pseudo_inputs (torch.Tensor): Initial pseudo-inputs of shape (K, input_dim).
            encoder (torch.nn.Module): Encoder that takes an input and returns (mean, log_var)
                for the approximate posterior q(z|x).
        """
        super(VampPrior, self).__init__()
        # Learnable pseudo-inputs; these are optimized during training.
        self.pseudo_inputs = torch.nn.Parameter(pseudo_inputs)
        self.encoder = encoder

    def get_mixture(self):
        """
        Constructs the VampPrior as a uniform mixture of q(z|u_k) for each pseudo-input u_k.
        """
        # Compute the approximate posterior parameters for each pseudo-input.
        # Expected output shapes: means and log_vars: (K, latent_dim)
        K = self.pseudo_inputs.size(0)
        # Create a dummy categorical argument; unsqueeze to have shape (K, 1)
        dummy_batch = torch.zeros(K, dtype=torch.long, device=self.pseudo_inputs.device).unsqueeze(1)
        means, log_vars, _ = self.encoder(self.pseudo_inputs, dummy_batch)
        scales = torch.sqrt(torch.exp(log_vars))
        component_dist = dist.Independent(dist.Normal(means, scales), 1)
        mixture_weights = torch.ones(K, device=self.pseudo_inputs.device) / K
        mixture = dist.MixtureSameFamily(
            dist.Categorical(mixture_weights),
            component_dist
        )
        return mixture

    def log_prob(self, z):
        """
        Computes log p(z) under the VampPrior.
        """
        return self.get_mixture().log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        """
        Draws samples from the VampPrior.
        """
        return self.get_mixture().sample(sample_shape)


class MixtureDirichlet(dist.TorchDistribution):
    """
    Mixture of Dirichlet distributions parameterized by mixture weights and concentration parameters."
    """
    arg_constraints = {
        "mixture_weights": dist.constraints.simplex,  # shape: batch_shape + (B,)
        "concentration": dist.constraints.positive,  # shape: batch_shape + (B, K)
    }
    support = dist.constraints.simplex  # each sample is a simplex over K categories
    has_rsample = False

    def __init__(
        self,
        mixture_weights: torch.Tensor,
        concentration: torch.Tensor,
        validate_args=None,
    ):
        """
        mixture_weights: Tensor of shape batch_shape + (B,), with each row summing to 1.
        concentration: Tensor of shape batch_shape + (B, K), where K is the number of categories.
        """
        self.mixture_weights = mixture_weights
        # Clamp concentrations to ensure positivity.
        self.concentration = torch.clamp(concentration, min=1e-3)
        # Determine batch shape, B, and K.
        batch_shape = self.mixture_weights.shape[:-1]
        self.B = self.mixture_weights.size(-1)
        self.K = self.concentration.size(-1)
        event_shape = (self.K,)
        super(MixtureDirichlet, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Returns a sample of shape: sample_shape + batch_shape + (K,).
        For each batch element, first sample a mixture component, then sample from the corresponding Dirichlet.
        """
        # Create categorical for mixture weights.
        cat = dist.Categorical(self.mixture_weights)
        # Sample mixture indices; shape: sample_shape + batch_shape.
        mixture_idx = cat.sample(sample_shape)
        full_shape = mixture_idx.shape  # sample_shape + batch_shape

        # Expand concentration to shape: sample_shape + batch_shape + (B, K).
        target_shape = sample_shape + self.concentration.shape
        expanded_concentration = self.concentration.expand(target_shape)

        # Flatten the sample and batch dimensions.
        flat_shape = (-1, self.B, self.K)
        flat_concentration = expanded_concentration.reshape(flat_shape)
        flat_idx = mixture_idx.reshape(-1)  # shape: (num_samples,)

        # Select the concentration parameters corresponding to the sampled mixture index.
        selected_concentration = flat_concentration[
            torch.arange(flat_idx.size(0)), flat_idx
        ]

        # Sample from the Dirichlet for each sample.
        flat_samples = dist.Dirichlet(selected_concentration).sample()
        # Reshape to sample_shape + batch_shape + (K,).
        return flat_samples.reshape(full_shape + (self.K,))

    def log_prob(self, value):
        device = value.device  # get the device from input tensor
        
        # Move tensors explicitly to the same device
        value_expanded = value.unsqueeze(-2).to(device)
        expanded_concentration = self.concentration.expand(
            value.shape[:-1] + (self.B, self.K)
        ).to(device)
        
        d = dist.Dirichlet(expanded_concentration)
        
        component_log_probs = d.log_prob(
            value_expanded.expand(expanded_concentration.shape)
        )
        
        expanded_weights = self.mixture_weights.expand(value.shape[:-1] + (self.B,)).to(device)
        mixture_log = torch.log(expanded_weights)
        
        return torch.logsumexp(mixture_log + component_log_probs, dim=-1)

    def score_parts(self, value):
        # Compute log probability.
        lp = self.log_prob(value)
        # Return dummy zeros for the score function and entropy terms.
        zeros = torch.zeros_like(lp)
        return lp, zeros, zeros

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
