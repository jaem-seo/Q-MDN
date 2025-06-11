import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt
import sys
from logistic import *

try:
    seed = int(sys.argv[1])
except:
    seed = 0

# Set seed
print("Seed:", seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set settings
lr = 0.005
n_comp, n_neuron = 5, 5
epochs, batch = 100, 64

# Gaussian function
def gaussian(y, mu, sigma):
    out = np.exp(-0.5 * (y - mu)**2 / sigma**2) / sigma / np.sqrt(2 * np.pi)
    return out

# Get logistic map dataset
xs, ys = [], []
x0, n, burn = 0.5, 100, 5
rs = np.arange(2.5, 4., 0.01)
for r in rs:
    xs += list(r * np.ones(n))
    ys += list(logistic_map(r, x0, n, discard=burn))

x_t = torch.tensor(xs, dtype=torch.float32)
y_t = torch.tensor(ys, dtype=torch.float32)
N = len(x_t)

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# MDN model
class ClassicalMDN(nn.Module):
    def __init__(self, in_dim=1, n_comp=n_comp, n_neuron=n_neuron):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, n_neuron),
            nn.Tanh(),
        )
        
        self.pi_head    = nn.Linear(n_neuron, n_comp, bias=True)
        self.mu_head    = nn.Linear(n_neuron, n_comp, bias=True)
        self.sigma_head = nn.Linear(n_neuron, n_comp, bias=True)
        self.log_sigma_offset = nn.Parameter(torch.zeros(n_comp))

    def forward(self, x):
        x = x.unsqueeze(1)          # (B,1)
        h = self.net(x)
        
        pi_logits = self.pi_head(h)           # (B,n_comp)
        pi        = F.softmax(pi_logits, dim=-1)

        mu        = self.mu_head(h)           # (B,n_comp)

        log_sigma =  self.sigma_head(h) + self.log_sigma_offset
        sigma     = torch.exp(log_sigma)

        return pi.squeeze(1), mu.squeeze(1), sigma.squeeze(1)

# Set model
mdn = ClassicalMDN()
print("Classical-MDN params :", n_params(mdn))      # ➜ 28

# Set optimizer
opt = torch.optim.Adam(mdn.parameters(), lr=lr)

# Set loss
def mdn_nll(pi, mu, sigma, y):
    y  = y.unsqueeze(1)                              # (B,1)
    gauss = torch.exp(-0.5*((y-mu)/sigma)**2) / (sigma*torch.sqrt(torch.tensor(2*np.pi)))
    pdf   = (pi * gauss).sum(dim=1) + 1e-12
    return -torch.log(pdf).mean()

# Train model
for epoch in range(epochs+1):
    
    perm = torch.randperm(N)
    epoch_nll = 0.
    for i in range(0, N, batch):
        idx = perm[i:i+batch]
        pi, mu, sigma = mdn(x_t[idx])
        loss = mdn_nll(pi, mu, sigma, y_t[idx])
        
        if epoch > 0:
            opt.zero_grad(); loss.backward(); opt.step()
        epoch_nll += loss.item()*len(idx)
        
    print(f"Ep {epoch:3d}  NLL={epoch_nll/N:.3f}")
    
    # Checkpoints
    if epoch % 10 == 0:

        yref = np.linspace(0, 1, 101)
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(2, 4))
        
        with torch.no_grad():
            xtest   = np.array([2.5, 3., 3.5, 3.95])
            pis, mus, sigmas = mdn(torch.tensor(xtest, dtype=torch.float32))
            
            for j in range(len(xtest)):
                p = 0.
                for i, pi in enumerate(pis[j].numpy()):
                    p += pi * gaussian(yref, mus[j,i].numpy(), sigmas[j,i].numpy())
                
                ys_ = np.random.choice(yref, size=100, replace=True, p=p/np.sum(p))
                
                axs[j].plot(yref, p, lw=2)
                axs[j].scatter(ys_, 1 + 0.05*np.random.randn(len(ys_)), c='b', s=2, alpha=0.2)
                axs[j].set_xlim([-0.2, 1.2])
                axs[j].set_ylim([0, 1.2])
                axs[j].spines['right'].set_visible(False)
                axs[j].spines['top'].set_visible(False)
                
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(f'mdn_seed{seed}_ep{epoch}.svg')

        torch.save(mdn.state_dict(), f'mdn_model_seed{seed}_ep{epoch}')
