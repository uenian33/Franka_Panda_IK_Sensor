import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, in_dim=7, out_dim=8, n_hidden=8):
        super(DenseNet, self).__init__()

        self.l1 = nn.Linear(in_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden+in_dim, n_hidden)
        self.l3 = nn.Linear(n_hidden+in_dim, n_hidden)
        self.l4 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(0.01) 


    def forward(self, state):
        a = F.relu(self.dropout(self.l1(state)))

        a = torch.cat([a, state], 1)
        a = F.relu(self.dropout(self.l2(a)))

        a = torch.cat([a, state], 1)
        a = F.relu(self.l3(a))

        a = self.l4(a)

        return a
    def sample(self, x):
        return self.forward(x)

class SelectionNet(nn.Module):
    """docstring for SimpleNet"""
    def __init__(self, in_dim=6, out_dim=7, hidden_dim=128):
        super().__init__()

         # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pred1 = nn.Linear(hidden_dim, out_dim)
        self.pred2 = nn.Linear(hidden_dim, out_dim)
        self.pred3 = nn.Linear(hidden_dim, out_dim)
        self.pred4 = nn.Linear(hidden_dim, out_dim)
        self.pred5 = nn.Linear(hidden_dim, out_dim)
        self.pred6 = nn.Linear(hidden_dim, out_dim)

        self.mask = nn.Linear(hidden_dim, out_dim)
        self.mask_acti = nn.Softmax()
    
    def forward(self, x):
        x = self.encoder(x)
        out_mask = self.mask(x)
        out_mask = self.mask_acti(out_mask)

        out = self.encoder(x)
        return out

    def sample(self, x):
        return self.forward(x)


class SimpleNet(nn.Module):
    """docstring for SimpleNet"""
    def __init__(self, in_dim=6, out_dim=7, hidden_dim=128):
        super().__init__()

         # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        out = self.encoder(x)
        return out

    def sample(self, x):
        return self.forward(x)


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)
