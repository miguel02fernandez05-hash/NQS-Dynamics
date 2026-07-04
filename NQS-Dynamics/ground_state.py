# ground_state_deuteron.py

import torch
import time
from torch import nn
from torch.autograd import grad
import numpy as np
from tqdm import tqdm
from torch.nn import Module
import math
from torch.func import functional_call, vmap, grad as func_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.set_default_dtype(torch.float64)

class DeuteronNQS(nn.Module):
    def __init__(self, W1_amp, B1_amp, W2_amp, B2_amp, W3_amp,
                 W1_phase, B1_phase, W2_phase, B2_phase, W3_phase,
                 Nin, Nhid, Nout, x0, y0, z0):
        super(DeuteronNQS, self).__init__()

        # Amplitude network.
        self.amp_lc1 = nn.Linear(in_features=Nin, out_features=Nhid, bias=True)
        self.amp_lc2 = nn.Linear(
            in_features=Nhid, out_features=Nhid, bias=True)
        self.amp_lc3 = nn.Linear(
            in_features=Nhid, out_features=Nout, bias=False)

        # Phase network.
        self.phase_lc1 = nn.Linear(
            in_features=Nin, out_features=Nhid, bias=True)
        self.phase_lc2 = nn.Linear(
            in_features=Nhid, out_features=Nhid, bias=True)
        self.phase_lc3 = nn.Linear(
            in_features=Nhid, out_features=Nout, bias=False)

        # Activation function.
        self.actfun = nn.Sigmoid()
        self.register_buffer('potential_center', torch.tensor([x0, y0, z0], dtype=torch.float64))
    
        # Exponential decay parameter.
        # 0.23 initial value of the exponential decay parameter, obtained by solving Schrodinger far away, where V(r)~0.
        self.alpha = nn.Parameter(torch.tensor([0.23], dtype=torch.float64))

        # Gamma parameter to boost the wavefunction and apply a kick.
        # Initialized as 0 for the g.s.
        self.register_buffer('gamma', torch.tensor([0.0], dtype=torch.float64))

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
        r2 = torch.sum((x-self.potential_center)**2, dim=1, keepdim=True)
        r = torch.sqrt(r2+1e-8)

        # Amplitude network.
        amplitude1 = self.actfun(self.amp_lc1(x))  # First hidden layer.
        amplitude2 = self.actfun(self.amp_lc2(amplitude1))
        # Envelope.
        r_aux=torch.sqrt(r2+0.5**2)-0.5
        amplitude=self.amp_lc3(amplitude2)-self.alpha*r_aux

        # Phase network.
        phase1 = self.actfun(self.phase_lc1(x))
        phase2 = self.actfun(self.phase_lc2(phase1))
        phase = self.phase_lc3(phase2)+self.gamma*r2

        return torch.complex(amplitude, phase)

    # Metropolis thermalization.
    def thermalization(self, walkers: torch.Tensor, step_size: float) -> torch.Tensor:
        walkers = walkers.detach().requires_grad_(False)
        val_old = self(walkers).real
        for _ in range(500):
            walkers_prim = walkers.clone()
            walkers_prim += torch.randn_like(walkers_prim)*step_size

            val_new = self(walkers_prim).real
            ratio = 2*(val_new-val_old)
            u = torch.log(torch.rand_like(walkers[:, 0:1]))

            condition = u < ratio
            walkers = torch.where(condition.expand_as(
                walkers), walkers_prim, walkers)
            val_old = torch.where(condition, val_new, val_old)
        return walkers

    def metropolis_step(self, walkers: torch.Tensor, step_size: float, N_acc: int = 0, steps_per_iteration: int = 100):
        walkers = walkers.detach().requires_grad_(False)
        aux = torch.zeros(1, device=walkers.device)
        val_old = self(walkers).real
        for _ in range(steps_per_iteration):
            walkers_prim = walkers.clone()
            walkers_prim += torch.randn_like(walkers_prim)*step_size

            val_new = self(walkers_prim).real

            ratio = 2*(val_new-val_old)
            u = torch.log(torch.rand_like(walkers[:, 0:1]))

            condition = u < ratio
            walkers = torch.where(condition.expand_as(
                walkers), walkers_prim, walkers)
            val_old = torch.where(condition, val_new, val_old)
            aux += condition.float().sum()
        N_acc = N_acc+aux
        return walkers, N_acc


def local_energy(net, walkers, config):
    kin_prefactor = config['kin_prefactor']
    C_10 = config['C_10']
    R_0 = config['R_0']
    center_potential = config['center_potential']

    walkers.requires_grad_(True)
    logpsi = net(walkers)

    logpsi_r = logpsi.real
    logpsi_i = logpsi.imag

    # First derivative. In 3D, grad gives (dpsi/dx, dpsi/dy, dpsi/dz).
    dlogpsi_r_dx = grad(outputs=logpsi_r, inputs=walkers,
                        grad_outputs=torch.ones_like(logpsi_r), create_graph=True)[0]
    dlogpsi_i_dx = grad(outputs=logpsi_i, inputs=walkers,
                        grad_outputs=torch.ones_like(logpsi_i), create_graph=True)[0]
    dlogpsi_dx = torch.complex(dlogpsi_r_dx, dlogpsi_i_dx)  # (n_walkers,3).

    # (n_walkers,1), each element is (d_dx)^2+(d_dy)^2+(d_dz)^2.
    grad_sq = torch.sum(dlogpsi_dx**2, dim=1, keepdim=True)

    # Second derivative.
    laplacian_r = torch.zeros_like(logpsi_r)
    laplacian_i = torch.zeros_like(logpsi_i)
    # Loop over x, y and z, avoiding cross-derivatives. With dim inside grad, we derive with respect to the selected dimension.
    # Actually maybe not faster than computing all derivatives, without the for.
    for dim in range(3):
        # Outputs to get only the first derivative with respect to dim. Inputs normal, computes cross derivatives too. [0][:, dim:dim+1] to avoid extracting cross derivatives.
        d2_r = grad(outputs=dlogpsi_r_dx[:, dim], inputs=walkers,
                    grad_outputs=torch.ones_like(dlogpsi_r_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_r += d2_r

        d2_i = grad(outputs=dlogpsi_i_dx[:, dim], inputs=walkers,
                    grad_outputs=torch.ones_like(dlogpsi_i_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
        laplacian_i += d2_i

    laplacian = torch.complex(laplacian_r, laplacian_i)

    KL = -kin_prefactor*(laplacian+grad_sq)

    r2 = torch.sum((walkers-center_potential)**2, dim=1, keepdim=True)

    UL = C_10*torch.exp(-r2/R_0**2)
    EL = KL+UL

    return EL, UL, KL, logpsi


# Stochastic Reconfiguration.
# .backward() is defined to work on the loss function (on a scalar). If we wanted to use .backward(), we would have
# to use a for and loop over all the walkers in order to build the quantum geometric tensor. Instead, we can use
# torch.func vmap.
# vmap takes creates a vectorizing map so that we can perform operations on many walkers simultaneously.
# parameters_to_vector and vector_to_parameters are the equivalents to ravel_pytree and unravel in PyTorch.
# They take the weights matrices and the bias arrays and transform it into a single array and viceversa.
# functional_call is to ignore the current weights of the network and use whatever weights I input (the ones from SR).

def sr(net, walkers, EL, config):

    n_walkers = config['n_walkers']
    epsilon = config['epsilonSR'] 

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

    flattened_real_grads = []
    flattened_imag_grads = []

    # Loop through every layer's gradient in the dictionary (jac_real_dict).
    for j_real in jac_real_dict.values():
        # Reshape into [n_walkers, whatever number needed to fit all].
        flat_j_real = j_real.view(n_walkers, -1)
        flattened_real_grads.append(flat_j_real)

    for j_imag in jac_imag_dict.values():
        flat_j_imag = j_imag.view(n_walkers, -1)
        flattened_imag_grads.append(flat_j_imag)

    # Concatenate them in size [n_walkers, whatever number needed to fit all].
    jac_real = torch.cat(flattened_real_grads, dim=1)
    jac_imag = torch.cat(flattened_imag_grads, dim=1)

    jac = torch.complex(jac_real, jac_imag)

    # O-<O>.
    jac_mean = jac.mean(dim=0, keepdim=True)
    O_prim = jac-jac_mean

    # Quantum geometric tensor.
    S = (O_prim.mH@O_prim).real/n_walkers    # .mH is matrix Hermitian.
    S = S+epsilon*torch.eye(S.shape[0], device=S.device)

    # Energy gradient.
    EL_centered = EL_det-EL_det.mean()
    f = 2*(EL_centered.conj().T@O_prim).real.squeeze(0)/n_walkers
    # f = 2*(O_prim.mH@EL_centered).real.squeeze(0)/n_walkers

    # Solve S*dp=f with Cholesky.
    L = torch.linalg.cholesky(S)
    dp = torch.cholesky_solve(f.unsqueeze(1), L).squeeze(1)

    return dp


def train_ground_state(net, optimizer, config, X_train, plot_callback=None):
    device = config['device']
    n_walkers = config['n_walkers']
    pretraining_epochs = config['pretraining_epochs']
    epochs = config['epochs']
    Nx, h, dV = config['Nx'], config['h'], config['dV']
    center_potential = config['center_potential']
    kin_prefactor = config['kin_prefactor']
    C_10, R_0 = config['C_10'], config['R_0']
    step_size = config['step_size_gs']
    steps_per_iteration = config['steps_per_iteration_gs']

    X_det = X_train.clone().detach()

    # Only used in case of pretraining, but pretraining_epochs=0.
    def loss_fn():
        log_psi = net(X_train)
        psi = torch.exp(log_psi)
        psi2 = psi.abs()**2

        psi_r = psi.real
        psi_i = psi.imag

        dpsi_r_dx = grad(outputs=psi_r, inputs=X_train,
                         grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
        dpsi_i_dx = grad(outputs=psi_i, inputs=X_train,
                         grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]

        laplacian_r = torch.zeros_like(psi_r)
        laplacian_i = torch.zeros_like(psi_i)

        for dim in range(3):
            d2_r = grad(outputs=dpsi_r_dx[:, dim], inputs=X_train,
                        grad_outputs=torch.ones_like(dpsi_r_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
            laplacian_r += d2_r

            d2_i = grad(outputs=dpsi_i_dx[:, dim], inputs=X_train,
                        grad_outputs=torch.ones_like(dpsi_i_dx[:, dim]), create_graph=True)[0][:, dim:dim+1]
            laplacian_i += d2_i

        laplacian = torch.complex(laplacian_r, laplacian_i)

        K_integrand = -kin_prefactor*(psi.conj()*laplacian).real

        r2_mesh = torch.sum((X_det-center_potential)**2, dim=1, keepdim=True)
        U_integrand = C_10*torch.exp(-r2_mesh/R_0**2)*psi2

        N = torch.sum(psi2)*dV
        U = torch.sum(U_integrand)*dV/N
        K = torch.sum(K_integrand)*dV/N
        E = U+K

        return E, U, K, psi, psi/torch.sqrt(N)

    aux_epochs = pretraining_epochs+epochs
    eval_interval = 100
    num_evals = sum(1 for i in range(epochs) if (
        i == epochs-1) or (i % eval_interval == 0))

    history = {
        'loss_accum': torch.zeros(aux_epochs, device=device),
        'U_accum': torch.zeros(aux_epochs, device=device),
        'K_accum': torch.zeros(aux_epochs, device=device),
        'r_walkers_accum': torch.zeros(aux_epochs, device=device),
        'mesh_epochs': torch.zeros(num_evals, device=device),
        'r_mesh_accum': torch.zeros(num_evals, device=device),
        'r2_mesh_accum': torch.zeros(num_evals, device=device),
        'r2_walkers_accum': torch.zeros(aux_epochs, device=device),
        'r_walkers_err_accum': torch.zeros(aux_epochs, device=device),
        'r2_walkers_err_accum': torch.zeros(aux_epochs, device=device),
        'r_mesh_err_accum': torch.zeros(num_evals, device=device),
        'r2_mesh_err_accum': torch.zeros(num_evals, device=device),
        'acceptance': 0
    }

    eval_idx = 0

    X_train.requires_grad_(True)

    for i in tqdm(range(pretraining_epochs), desc="Pretraining the NQS..."):
        optimizer.zero_grad()
        loss0.backward()       
        optimizer.step()     

        loss0, U, K, psi, psi_normalized = loss_fn()

        history['loss_accum'][i] = loss0.item()
        history['U_accum'][i] = U.item()
        history['K_accum'][i] = K.item()

        if plot_callback and (i == pretraining_epochs-1) or (i % 50 == 0):
            plot_callback(i, psi, psi_normalized,
                          history['loss_accum'], history['U_accum'], history['K_accum'])

    # Initialize walkers. Now Gaussian distrib.
    walkers = torch.empty(n_walkers, 3, device=device).normal_(mean=0, std=0.5)

    N_acc = 0
    N_total = 0

    walkers = net.thermalization(walkers, step_size)

    for i in tqdm(range(epochs), desc="Training the NQS..."):

        initial_lr = config['initial_lr']
        final_lr = config['final_lr']
        decay_factor = (final_lr/initial_lr)**(i/epochs)
        lr = initial_lr*decay_factor

        walkers, N_acc = net.metropolis_step(
            walkers, step_size, N_acc, steps_per_iteration)
        N_total += steps_per_iteration*n_walkers

        EL, UL, KL, logpsi_walkers = local_energy(net, walkers, config)

        E_mean, UL_mean, KL_mean = EL.mean(), UL.mean(), KL.mean()

        r_walkers_dist = torch.sqrt(
            torch.sum((walkers-center_potential)**2, dim=1))
        history['r_walkers_accum'][i +
                                   pretraining_epochs] = r_walkers_dist.mean().detach()
        history['r_walkers_err_accum'][i+pretraining_epochs] = r_walkers_dist.std().detach() /\
            math.sqrt(n_walkers)

        r2_walkers_sq = torch.sum((walkers-center_potential)**2, dim=1)
        r2_mean = r2_walkers_sq.mean().detach()
        r2_std = r2_walkers_sq.std().detach()
        history['r2_walkers_accum'][i +
                                    pretraining_epochs] = torch.sqrt(r2_mean)/2
        history['r2_walkers_err_accum'][i+pretraining_epochs] = (
            1.0/(4*torch.sqrt(r2_mean)))*(r2_std/math.sqrt(n_walkers))

        dp = sr(net, walkers, EL, config)
        
        with torch.no_grad():
            current_params = parameters_to_vector(net.parameters())
            new_params = current_params-lr*dp
            vector_to_parameters(new_params, net.parameters())

        history['loss_accum'][i+pretraining_epochs] = E_mean.real.detach()
        history['U_accum'][i+pretraining_epochs] = UL_mean.real.detach()
        history['K_accum'][i+pretraining_epochs] = KL_mean.real.detach()

        if (i == epochs-1) or (i % eval_interval == 0):
            with torch.no_grad():
                chunk_size = 5000000
                logpsi_list = []
                for start_idx in range(0, X_train.shape[0], chunk_size):
                    end_idx = min(start_idx+chunk_size, X_train.shape[0])
                    logpsi_list.append(net(X_train[start_idx:end_idx]))

                logpsi_mesh = torch.cat(logpsi_list, dim=0)
                psi_mesh = torch.exp(logpsi_mesh)
                norm_const = torch.sqrt(torch.sum(psi_mesh.abs()**2)*dV)
                psi_norm_mesh = psi_mesh/norm_const

                if plot_callback:
                    plot_callback(i+pretraining_epochs, psi_mesh, psi_norm_mesh,
                                  history['loss_accum'], history['U_accum'], history['K_accum'])

                # For error inherent to doing grid calculations, derived using taylor.
                psi2_unnorm = psi_mesh.abs()**2
                b_val = torch.sum(psi2_unnorm)*dV

                r_mesh_dist = torch.sqrt(
                    torch.sum((X_train-center_potential)**2, dim=1, keepdim=True))
                r2_mesh = torch.sum((X_train-center_potential)
                                    ** 2, dim=1, keepdim=True)

                a1_flat = (r_mesh_dist*psi2_unnorm).squeeze()
                a2_flat = (r2_mesh*psi2_unnorm).squeeze()
                b_flat = psi2_unnorm.squeeze()

                a1_val = torch.sum(a1_flat)*dV
                a2_val = torch.sum(a2_flat)*dV

                def laplacian_3d(f):
                    d2f_dx2 = (f[2:, 1:-1, 1:-1]-2*f[1:-1, 1:-1,
                                                     1:-1]+f[:-2, 1:-1, 1:-1])/(h**2)
                    d2f_dy2 = (f[1:-1, 2:, 1:-1]-2*f[1:-1, 1:-1,
                                                     1:-1]+f[1:-1, :-2, 1:-1])/(h**2)
                    d2f_dz2 = (f[1:-1, 1:-1, 2:]-2*f[1:-1, 1:-1,
                                                     1:-1]+f[1:-1, 1:-1, :-2])/(h**2)
                    return d2f_dx2+d2f_dy2+d2f_dz2

                a1_3d = a1_flat.view(Nx, Nx, Nx)
                a2_3d = a2_flat.view(Nx, Nx, Nx)
                b_3d = b_flat.view(Nx, Nx, Nx)

                delta_a1 = (h**5/24)*torch.sum(torch.abs(laplacian_3d(a1_3d)))
                delta_a2 = (h**5/24)*torch.sum(torch.abs(laplacian_3d(a2_3d)))
                delta_b = (h**5/24)*torch.sum(torch.abs(laplacian_3d(b_3d)))

                err_r_expected = (delta_a1/b_val)+(a1_val/(b_val**2))*delta_b
                err_r2_expected_sq = (delta_a2/b_val) +\
                    (a2_val/(b_val**2))*delta_b

                r_expected_val = a1_val/b_val
                r2_expected_val = torch.sqrt(a2_val/b_val)/2

                r2_mesh = torch.sum(
                    (X_train - center_potential)**2, dim=1, keepdim=True)
                r2_expected_val = torch.sqrt(
                    torch.sum(r2_mesh * (psi_norm_mesh.abs()**2)) * dV)/2

                history['mesh_epochs'][eval_idx] = i+pretraining_epochs
                history['r_mesh_accum'][eval_idx] = r_expected_val
                history['r_mesh_err_accum'][eval_idx] = err_r_expected
                history['r2_mesh_accum'][eval_idx] = r2_expected_val
                history['r2_mesh_err_accum'][eval_idx] = (
                    1.0/(4*torch.sqrt(a2_val/b_val)))*err_r2_expected_sq

                eval_idx += 1

    history['acceptance'] = int(N_acc.item())/N_total
    history['norm_const'] = norm_const

    return net, walkers, history
