import pennylane as qml, torch, numpy as np
import sys
from torch import nn, optim
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

try:
    seed = int(sys.argv[1])
except:
    seed = 0

# Set seed
print("Seed:", seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set settings
N = 20000
lr = 0.005
n_qubits, layers = 3, 4
epochs, batch = 100, 64

# Set peak parameters
interf_centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
slit_centers   = np.array([-1.5, 1.5])
interf_sigma = 0.15
slit_sigma   = 0.1
interf_alpha = np.array([0.125, 0.2, 0.35, 0.2, 0.125])
slit_alpha = np.array([0.5, 0.5])

# Gaussian function
def gaussian(y, mu, sigma):
    out = np.exp(-0.5 * (y - mu)**2 / sigma**2) / sigma / np.sqrt(2 * np.pi)
    return out

# Mixture weights (interference + classical)
def mixture_weights(x):
    x = np.atleast_1d(x)
    w_interf = (1 - x)[:, None] * interf_alpha              # (B,5)
    w_slit   = x[:, None]       * slit_alpha                # (B,2)
    return np.concatenate([w_interf, w_slit], axis=1)       # (B,7)

# Data generator
def sample_dataset(n_samples=10000, x_sampler=None, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    if x_sampler is None:
        x_vals = rng.random(n_samples)               # Uniform U[0,1]
    else:
        x_vals = x_sampler(n_samples)
    
    centers = np.concatenate([interf_centers, slit_centers])   # (7,)
    sigmas  = np.concatenate([
        np.full_like(interf_centers, interf_sigma),
        np.full_like(slit_centers,        slit_sigma)
    ])  # (7,)
    
    weights = mixture_weights(x_vals)                # (N,7)
    
    comp_choices = [
        rng.choice(7, p=w) for w in weights        
    ]
    comp_choices = np.array(comp_choices)            # (N,)
    
    y_vals = rng.normal(
        loc=centers[comp_choices],
        scale=sigmas[comp_choices]
    )
    return x_vals, y_vals

dev = qml.device("default.qubit", wires=n_qubits)

# Q-MDN cirquit and model
@qml.qnode(dev, interface="torch")
def circuit(x, weights, sample=False):
    qml.AngleEmbedding([x, x*np.pi/2], wires=list(range(n_qubits)))
    qml.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
    return qml.probs(wires=list(range(n_qubits)))          

def probs_to_logits(p, eps=1e-7):
    p = torch.clamp(p, eps, 1-eps)        # to avoid log(0)
    logit = torch.log(p[..., :-1]) - torch.log(p[..., -1:])
    return logit                          # (0, 1) to (-inf, inf)

class QMDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_p = nn.Parameter(0.05*torch.randn(layers, n_qubits, 3))
        self.weights_m = nn.Parameter(0.05*torch.randn(layers, n_qubits, 3))
        self.weights_s = nn.Parameter(0.05*torch.randn(layers, n_qubits, 3))

    def forward(self, x):
        pi        = torch.stack([circuit(xi, self.weights_p) for xi in x])
        pi        = pi[..., :-1] / pi[..., :-1].sum(axis=1, keepdim=True)

        mu_probs  = torch.stack([circuit(xi, self.weights_m) for xi in x])
        mu        = probs_to_logits(mu_probs)

        s_probs   = torch.stack([circuit(xi, self.weights_s) for xi in x])
        log_sigma = probs_to_logits(s_probs)
        sigma     = torch.exp(log_sigma)             # σ>0
        return pi, mu, sigma

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set model
model = QMDN()
print("Quantum-MDN params :", n_params(model))

# Set loss
def mdn_nll(pi, mu, sigma, y):
    y = y.unsqueeze(1)                               # (B,1)
    gauss = torch.exp(-0.5*((y-mu)/sigma)**2) / (sigma*torch.sqrt(torch.tensor(2*np.pi)))
    pdf   = (pi * gauss).sum(dim=1) + 1e-12
    return -torch.log(pdf).mean()

# Set optimizer
opt = optim.Adam(model.parameters(), lr=lr)

# Train model
for ep in range(epochs+1):
    xs, ys = sample_dataset(N)
    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)

    perm = torch.randperm(N)
    epoch_nll = 0.
    for i in range(0, N, batch):
        idx = perm[i:i+batch]
        pi, mu, sigma = model(x_t[idx])
        loss = mdn_nll(pi, mu, sigma, y_t[idx])

        if ep>0:
            opt.zero_grad(); loss.backward(); opt.step()
        epoch_nll += loss.item()*len(idx)
    
    print(f"Ep {ep:3d}  NLL={epoch_nll/N:.3f}")

    # Checkpoints
    if ep % 10 == 0:
        yref = np.linspace(-2.5, 2.5, 101)
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(2, 1.75))
        
        with torch.no_grad():
            xtest   = np.array([0.,1.])
            pis, mus, sigmas = model(torch.tensor(xtest, dtype=torch.float32))
            
            for j in range(len(xtest)):
                p = 0.
                for i, pi in enumerate(pis[j].numpy()):
                    p += pi * gaussian(yref, mus[j,i].numpy(), sigmas[j,i].numpy())
                
                ys_ = np.random.choice(yref, size=100, replace=True, p=p/np.sum(p))
                
                axs[j].plot(yref, p, lw=2)
                axs[j].scatter(ys_, 1 + 0.05*np.random.randn(len(ys_)), c='b', s=2, alpha=0.2)
                axs[j].set_xlim([-2.75, 2.75])
                axs[j].set_ylim([0, 1.2])
                axs[j].spines['right'].set_visible(False)
                axs[j].spines['top'].set_visible(False)
                
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(f'qmdn_lr{lr}_seed{seed}_ep{ep}.svg')
        
        torch.save(model.state_dict(), f'qmdn_model_lr{lr}_seed{seed}_ep{ep}')
