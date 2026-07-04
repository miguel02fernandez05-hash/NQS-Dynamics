# ground_state_QHO.py

import torch
from torch import nn
from torch.autograd import grad
import numpy as np
from tqdm import tqdm
from torch.func import functional_call, vmap, grad as func_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math

torch.set_default_dtype(torch.float64)

class ComplexHarmonicNQS(nn.Module):
    def __init__(self, Nin, Nout, Nhid, W1_amp, B1_amp, W2_amp, B2_amp, W3_amp, 
                       W1_phase, B1_phase, W2_phase, B2_phase, W3_phase, x0, y0, z0):
        super(ComplexHarmonicNQS, self).__init__()
        
        # Amplitude network.
        self.amp_lc1 = nn.Linear(in_features=Nin, out_features=Nhid, bias=True)
        self.amp_lc2 = nn.Linear(in_features=Nhid, out_features=Nhid, bias=True)
        self.amp_lc3 = nn.Linear(in_features=Nhid, out_features=Nout, bias=False)
        
        # Phase network.
        self.phase_lc1 = nn.Linear(in_features=Nin, out_features=Nhid, bias=True)
        self.phase_lc2 = nn.Linear(in_features=Nhid, out_features=Nhid, bias=True)
        self.phase_lc3 = nn.Linear(in_features=Nhid, out_features=Nout, bias=False)
        
        # Activation function.
        self.actfun = nn.Sigmoid() 
        self.apply_kick = False 
        self.register_buffer('kick_momentum', torch.zeros(Nin))
        # Envelope center as a parameter.
        self.envelope_center = nn.Parameter(torch.tensor([x0, y0, z0], dtype=torch.float64))
        # Fixed envelope center (commented).
        # self.register_buffer('envelope_center', torch.tensor([x0, y0, z0], dtype=torch.float64))

        # Loading the initial value of the weight matrix and bias vector.
        with torch.no_grad():
            self.amp_lc1.weight = nn.Parameter(W1_amp)
            self.amp_lc1.bias = nn.Parameter(B1_amp)
            self.amp_lc2.weight = nn.Parameter(W2_amp)
            self.amp_lc2.bias = nn.Parameter(B2_amp)
            self.amp_lc3.weight = nn.Parameter(W3_amp)
            
            self.phase_lc1.weight = nn.Parameter(W1_phase)
            self.phase_lc1.bias = nn.Parameter(B1_phase)
            self.phase_lc2.weight = nn.Parameter(W2_phase)
            self.phase_lc2.bias = nn.Parameter(B2_phase)
            self.phase_lc3.weight = nn.Parameter(W3_phase)

    def forward(self, x):
        r2 = torch.sum((x - self.envelope_center)**2, dim=1, keepdim=True)
        
        # Amplitude network.
        amplitude1 = self.actfun(self.amp_lc1(x))
        amplitude2 = self.actfun(self.amp_lc2(amplitude1))
        amplitude = self.amp_lc3(amplitude2) - 0.04 * r2

        # Phase network.
        phase1 = self.actfun(self.phase_lc1(x))
        phase2 = self.actfun(self.phase_lc2(phase1))
        phase = self.phase_lc3(phase2)
        
        if self.apply_kick: 
            phase = phase + torch.sum(self.kick_momentum * x, dim=1, keepdim=True)
        
        return torch.complex(amplitude, phase)
    
    # Metropolis thermalization.
    def thermalization(self, walkers: torch.Tensor, step_size: float) -> torch.Tensor:
        walkers = walkers.detach().requires_grad_(False)
        val_old = self(walkers).real
        for _ in range(500):
            walkers_prim = walkers.clone()
            walkers_prim += torch.randn_like(walkers_prim) * step_size
            
            val_new = self(walkers_prim).real
            ratio = 2 * (val_new - val_old)
            u = torch.log(torch.rand_like(walkers[:, 0:1]))
            condition = u < ratio
            walkers = torch.where(condition.expand_as(walkers), walkers_prim, walkers)
            val_old = torch.where(condition, val_new, val_old)
        return walkers

    def metropolis_step(self, walkers: torch.Tensor, step_size: float, N_acc: int = 0, steps_per_iteration: int = 50):
        walkers = walkers.detach().requires_grad_(False)
        aux = torch.zeros(1, device=walkers.device)
        val_old = self(walkers).real
        for _ in range(steps_per_iteration):
            walkers_prim = walkers.clone()
            walkers_prim += torch.randn_like(walkers_prim) * step_size
            
            val_new = self(walkers_prim).real
            ratio = 2 * (val_new - val_old)
            u = torch.log(torch.rand_like(walkers[:, 0:1]))

            condition = u < ratio
            walkers = torch.where(condition.expand_as(walkers), walkers_prim, walkers)
            val_old = torch.where(condition, val_new, val_old)
            aux += condition.float().sum()
        N_acc = N_acc + aux
        return walkers, N_acc

def local_energy(net, walkers, freqs_sq, shifts):
    walkers.requires_grad_(True)
    logpsi = net(walkers)
    
    logpsi_r = logpsi.real
    logpsi_i = logpsi.imag
    
    # First derivative. In 3D, grad gives (dpsi/dx, dpsi/dy, dpsi/dz).
    dlogpsi_r_dx = grad(outputs=logpsi_r, inputs=walkers, grad_outputs=torch.ones_like(logpsi_r), create_graph=True)[0]
    dlogpsi_i_dx = grad(outputs=logpsi_i, inputs=walkers, grad_outputs=torch.ones_like(logpsi_i), create_graph=True)[0]
    dlogpsi_dx = torch.complex(dlogpsi_r_dx, dlogpsi_i_dx)
    
    # (n_walkers,1), each element is (d_dx)^2+(d_dy)^2+(d_dz)^2.
    grad_sq = torch.sum(dlogpsi_dx**2, dim=1, keepdim=True)
    
    # Second derivative.
    laplacian_r = torch.zeros_like(logpsi_r)
    laplacian_i = torch.zeros_like(logpsi_i)
    
    # Loop over x, y and z, avoiding cross-derivatives. With dim inside grad, we derive with respect to the selected dimension.
    # Actually maybe not faster than computing all derivatives, without the for.
    for dim in range(3):
        # Outputs to get only the first derivative with respect to dim. Inputs normal, computes cross derivatives too. [0][:, dim:dim+1] to avoid extracting cross derivatives.
        d2_r = grad(outputs=dlogpsi_r_dx[:, dim], inputs=walkers, grad_outputs=torch.ones_like(dlogpsi_r_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_r += d2_r
        d2_i = grad(outputs=dlogpsi_i_dx[:, dim], inputs=walkers, grad_outputs=torch.ones_like(dlogpsi_i_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_i += d2_i
    
    laplacian = torch.complex(laplacian_r, laplacian_i)
    KL = -0.5 * (laplacian + grad_sq)
    
    UL = 0.5 * torch.sum(freqs_sq * (walkers - shifts)**2, dim=1, keepdim=True)
    EL = KL + UL
    
    return EL, UL, KL, logpsi

# This is only used in case of pretraining, but pretraining_epochs=0.
def loss_fn(net, X, X_det, dV, shifts, freqs_sq):    
    log_psi = net(X)
    psi = torch.exp(log_psi)
    psi2 = psi.abs()**2
    
    psi_r = psi.real
    psi_i = psi.imag
    
    dpsi_r_dx = grad(outputs=psi_r, inputs=X, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
    dpsi_i_dx = grad(outputs=psi_i, inputs=X, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]
    
    laplacian_r = torch.zeros_like(psi_r)
    laplacian_i = torch.zeros_like(psi_i)
    
    for dim in range(3):
        d2_r = grad(outputs=dpsi_r_dx[:, dim], inputs=X, grad_outputs=torch.ones_like(dpsi_r_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_r += d2_r
        d2_i = grad(outputs=dpsi_i_dx[:, dim], inputs=X, grad_outputs=torch.ones_like(dpsi_i_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_i += d2_i
        
    laplacian = torch.complex(laplacian_r, laplacian_i)
    
    K_integrand = -0.5 * (psi.conj() * laplacian).real
    U_integrand = 0.5 * torch.sum(freqs_sq * (X_det - shifts)**2, dim=1, keepdim=True) * psi2
    
    N = torch.sum(psi2) * dV    
    U = torch.sum(U_integrand) * dV / N
    K = torch.sum(K_integrand) * dV / N
    E = U + K
    
    return E, U, K, psi, psi / torch.sqrt(N)

# Stochastic Reconfiguration.
# .backward() is defined to work on the loss function (on a scalar). If we wanted to use .backward(), we would have
# to use a for and loop over all the walkers in order to build the quantum geometric tensor. Instead, we can use
# torch.func vmap.
# vmap takes creates a vectorizing map so that we can perform operations on many walkers simultaneously.
# parameters_to_vector and vector_to_parameters are the equivalents to ravel_pytree and unravel in PyTorch.
# They take the weights matrices and the bias arrays and transform it into a single array and viceversa.
# functional_call is to ignore the current weights of the network and use whatever weights I input (the ones from SR).

def sr(net, walkers, EL, n_walkers, epsilon):
    walkers_det = walkers.detach()
    EL_det = EL.detach()

    # Dictionary of the parameters for functional_call.
    params = dict(net.named_parameters())

    def compute_logpsi_real(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze().real
    def compute_logpsi_imag(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze().imag
    
    # Jacobian using vmap.
    # None so that same parameters for all. 0 to split in rows, each row is a walker.
    jac_real_fn = vmap(func_grad(compute_logpsi_real), in_dims=(None, 0))
    jac_imag_fn = vmap(func_grad(compute_logpsi_imag), in_dims=(None, 0))
    
    # Inputs of compute_logpsi_real.
    jac_real_dict = jac_real_fn(params, walkers_det)
    jac_imag_dict = jac_imag_fn(params, walkers_det)

    # Loop through every layer's gradient in the dictionary (jac_real_dict). Reshape into [n_walkers, whatever number needed to fit all].
    flattened_real_grads = [j_real.view(n_walkers, -1) for j_real in jac_real_dict.values()]
    flattened_imag_grads = [j_imag.view(n_walkers, -1) for j_imag in jac_imag_dict.values()]
    
    # Concatenate them in size [n_walkers, whatever number needed to fit all].
    jac_real = torch.cat(flattened_real_grads, dim=1)
    jac_imag = torch.cat(flattened_imag_grads, dim=1)
    jac = torch.complex(jac_real, jac_imag)

    # O-<O>.
    jac_mean = jac.mean(dim=0, keepdim=True)
    O_prim = jac - jac_mean

    # Quantum geometric tensor.
    S = (O_prim.mH @ O_prim).real / n_walkers
    S = S + epsilon * torch.eye(S.shape[0], device=S.device)

    EL_centered = EL_det - EL_det.mean()
    f = 2.0 * (EL_centered.conj().T @ O_prim).real.squeeze(0) / n_walkers

    # Solve S*dp=f with Cholesky.
    L = torch.linalg.cholesky(S)
    dp = torch.cholesky_solve(f.unsqueeze(1), L).squeeze(1)

    return dp

def train_ground_state(net, optimizer, X_train, coords_plot, freqs_sq, shifts, 
                       pretraining_epochs, epochs, n_walkers, dV, lr, target, wx, wy, wz, x0, y0, z0, device, Nx, epsilon, plot_callback=None):

    aux = pretraining_epochs + epochs
    loss_accum = torch.zeros(aux, device=device)
    U_accum = torch.zeros(aux, device=device)
    K_accum = torch.zeros(aux, device=device)

    X = X_train.clone()
    X_det = X.clone().detach()

    # Pretraining, but pretraining_epochs=0.
    for i in tqdm(range(pretraining_epochs), desc="Pretraining the NQS..."):    
        optimizer.zero_grad()
        loss0, U, K, psi, psi_normalized = loss_fn(net, X, X_det, dV, shifts, freqs_sq)
        loss0.backward()
        optimizer.step()
        
        loss_accum[i] = loss0.item()
        U_accum[i] = U.item()
        K_accum[i] = K.item()
        
        if (i == pretraining_epochs - 1) or (i % 50 == 0):
            if plot_callback:
                plot_callback(i, psi, psi_normalized, loss_accum, U_accum, K_accum)

    # Initialize walkers.
    walkers = torch.empty(n_walkers, 3, device=device).uniform_(-3.0, 3.0)
    
    walkers_init = walkers.detach().clone()

    walkers = net.thermalization(walkers, step_size=0.7)

    N_acc = 0
    N_total = 0
    step_size = 0.7
    steps_per_iteration = 50

    for i in tqdm(range(epochs), desc="Training the NQS..."):  
        walkers, N_acc = net.metropolis_step(walkers, step_size, N_acc, steps_per_iteration)
        N_total += steps_per_iteration * n_walkers
        
        EL, UL, KL, logpsi_walkers = local_energy(net, walkers, freqs_sq, shifts)
        
        E_mean = EL.mean()
        dp = sr(net, walkers, EL, n_walkers, epsilon)
        
        with torch.no_grad():
            current_params = parameters_to_vector(net.parameters())
            new_params = current_params - lr * dp
            vector_to_parameters(new_params, net.parameters())
        
        UL_mean = UL.mean()
        KL_mean = KL.mean()
        
        loss_accum[i+pretraining_epochs] = E_mean.real.detach()
        U_accum[i+pretraining_epochs] = UL_mean.real.detach()
        K_accum[i+pretraining_epochs] = KL_mean.real.detach()
        
        if (i == epochs - 1) or (i % 50 == 0):
           with torch.no_grad():
                chunk_size = 50000
                logpsi_list = []
                for start_idx in range(0, X_train.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, X_train.shape[0])
                    chunk = X_train[start_idx:end_idx]
                    logpsi_list.append(net(chunk))
                
                logpsi_mesh = torch.cat(logpsi_list, dim=0)
                psi_mesh = torch.exp(logpsi_mesh)
                norm_const = torch.sqrt(torch.sum(psi_mesh.abs()**2)*dV)
                psi_norm_mesh = psi_mesh / norm_const
                
                if plot_callback:
                    plot_callback(i+pretraining_epochs, psi_mesh, psi_norm_mesh, loss_accum, U_accum, K_accum)

    acceptance = int(N_acc.item()) / N_total
    print('\nAcceptance =', acceptance)        

    E_var = torch.var(EL).real.item()
    E_err = (torch.std(EL).real / math.sqrt(n_walkers)).item()

    return net, walkers, loss_accum, walkers_init, E_var, E_err