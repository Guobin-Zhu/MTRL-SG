from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network
    """
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = F.tanh
        self.outfn = F.tanh

    def forward(self, X: torch.Tensor):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.outfn(self.fc4(h3))
        return out

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize neural network layer weights and biases.
    """
    # Apply Kaiming (He) normal initialization to weights
    torch.nn.init.kaiming_normal_(layer.weight, std)
    
    # Initialize biases to a constant value
    torch.nn.init.constant_(layer.bias, bias_const)
    
    return layer


def normalize_embeds(embeds: nn.Embedding):
    """
    Normalize embedding vectors with added noise.
    """
    # Perform operations without tracking gradients
    with torch.no_grad():
        # Iterate through all embedding vectors
        for i in range(embeds.weight.size(0)):
            # Calculate current L2 norm of embedding vector
            _norm = torch.linalg.norm(embeds.weight[i])
            
            # Normalize and add noise: divide by (norm + random noise)
            # The noise magnitude is 20% of the original norm
            embeds.weight[i].div_(_norm + torch.randn_like(_norm) * 0.2 * _norm)


def orth_to(u: torch.Tensor, v: torch.Tensor):
    """
    Compute vector orthogonal to u that's a modified projection of v.
    """
    # Compute orthogonal projection: 
    # orth_vector = (u·u)v - (v·u)u  (Gram-Schmidt-like step)
    orth_vector = u.inner(u) * v - v.inner(u) * u
    
    # Safety check: verify orthogonality (should be near zero)
    assert u.inner(orth_vector).abs() < 1e-3
    
    # Add 20% Gaussian noise to the orthogonal vector
    return orth_vector + torch.randn_like(orth_vector) * 0.2 * orth_vector