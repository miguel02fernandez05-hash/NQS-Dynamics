# time_evolution_QHO.py

import torch
import numpy as np
import math
from tqdm import tqdm
from torch.func import vmap, grad as func_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ground_state_QHO import local_energy
from torch.func import functional_call

torch.set_default_dtype(torch.float64)

def tdvp(net, walkers, EL, n_walkers, epsilon):
    walkers_det = walkers.detach()
    EL_det = EL.detach()
    params = dict(net.named_parameters())
    
    def compute_logpsi_real(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze().real
    def compute_logpsi_imag(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze().imag
    
    # Jacobian using vmap (same as SR).
    jac_real_fn = vmap(func_grad(compute_logpsi_real), in_dims=(None, 0))
    jac_imag_fn = vmap(func_grad(compute_logpsi_imag), in_dims=(None, 0))
    
    jac_real_dict = jac_real_fn(params, walkers_det)
    jac_imag_dict = jac_imag_fn(params, walkers_det)

    # Loop through every layer's gradient in the dictionary.
    flattened_real_grads = [j_real.view(n_walkers, -1) for j_real in jac_real_dict.values()]
    flattened_imag_grads = [j_imag.view(n_walkers, -1) for j_imag in jac_imag_dict.values()]
    
    # Concatenate them in size [n_walkers, whatever number needed to fit all].
    jac_real = torch.cat(flattened_real_grads, dim=1)
    jac_imag = torch.cat(flattened_imag_grads, dim=1)
    jac = torch.complex(jac_real, jac_imag)

    # O-<O>.
    jac_mean = jac.mean(dim=0, keepdim=True)
    O_prim = jac - jac_mean

    # Now # F = Im(<O* E_L>).
    F = (O_prim.mH @ (EL_det - EL_det.mean())).imag.squeeze(1) / n_walkers
    S = (O_prim.mH @ O_prim).real / n_walkers
    aux = 1 + torch.abs(F)
    S = S + epsilon * torch.diag(aux)

    L = torch.linalg.cholesky(S)
    dp = torch.cholesky_solve(F.unsqueeze(1), L).squeeze(1)

    return dp

# RK2 and RK4 functions, although RK2 function is the one used.
def step_time_rk2(net, walkers, dt, n_walkers, freqs_sq, shifts_TE, step_size, steps_per_iteration, epsilon):
    walkers1, _ = net.metropolis_step(
        walkers,
        step_size,
        N_acc=0,
        steps_per_iteration=steps_per_iteration
    )

    EL1, _, _, _ = local_energy(net, walkers1, freqs_sq, shifts_TE)

    k1 = tdvp(net, walkers1, EL1, n_walkers, epsilon)

    with torch.no_grad():
        p0 = parameters_to_vector(net.parameters())

    with torch.no_grad():
        p_temp = p0 + dt * k1
        vector_to_parameters(p_temp, net.parameters())

    walkers2, _ = net.metropolis_step(
        walkers1,
        step_size,
        N_acc=0,
        steps_per_iteration=steps_per_iteration
    )

    EL2, _, _, _ = local_energy(net, walkers2, freqs_sq, shifts_TE)

    k2 = tdvp(net, walkers2, EL2, n_walkers, epsilon)

    with torch.no_grad():
        p_new = p0 + 0.5 * dt * (k1 + k2)
        vector_to_parameters(p_new, net.parameters())

    walkersfinal, _ = net.metropolis_step(
        walkers2,
        step_size,
        N_acc=0,
        steps_per_iteration=steps_per_iteration
    )

    ELfinal, _, _, _ = local_energy(net, walkersfinal, freqs_sq, shifts_TE)

    return walkersfinal, ELfinal


def step_time_rk4(net, walkers, dt, n_walkers, freqs_sq, shifts_TE, step_size, steps_per_iteration, epsilon):
    # Current state.
    walkers1, _=net.metropolis_step(walkers, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL1, _, _, _=local_energy(net, walkers1, freqs_sq, shifts_TE)
    
    # K1.
    k1=tdvp(net, walkers1, EL1, n_walkers, epsilon)
    
    # Save original parameters.
    with torch.no_grad():
        p0=parameters_to_vector(net.parameters())
    
    # K2.
    with torch.no_grad():
        p_temp=p0+0.5*dt*k1
        vector_to_parameters(p_temp, net.parameters())
    
    walkers2, _=net.metropolis_step(walkers1, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL2, _, _, _=local_energy(net, walkers2, freqs_sq, shifts_TE)
    
    k2 = tdvp(net, walkers2, EL2, n_walkers, epsilon)
    
    # K3.
    with torch.no_grad():
        p_temp=p0+0.5*dt*k2
        vector_to_parameters(p_temp, net.parameters())
    
    walkers3, _=net.metropolis_step(walkers2, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL3, _, _, _=local_energy(net, walkers3, freqs_sq, shifts_TE)
    
    k3=tdvp(net, walkers3, EL3, n_walkers, epsilon)

    # K4.
    with torch.no_grad():
        p_temp=p0+dt*k3
        vector_to_parameters(p_temp, net.parameters())

    walkers4, _=net.metropolis_step(walkers3, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL4, _, _, _=local_energy(net, walkers4, freqs_sq, shifts_TE)
    
    k4=tdvp(net, walkers4, EL4, n_walkers, epsilon)

    # Combine.
    with torch.no_grad():
        p_new=p0+(dt/6)*(k1+2*k2+2*k3+k4)
        vector_to_parameters(p_new, net.parameters())

    walkersfinal, _=net.metropolis_step(walkers4, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    ELfinal, _, _, _=local_energy(net, walkersfinal, freqs_sq, shifts_TE)

    return walkersfinal, ELfinal

# Exact evolution of the QHO, for fidelity.
def exact_evolution_log(walkers, t_val, shifts_TE, shifts_GS, p0_kick, freqs):
    t = torch.tensor(t_val, device=walkers.device, dtype=torch.float64)
    
    x_prime_0 = shifts_GS - shifts_TE
    
    x_c_prime = x_prime_0 * torch.cos(freqs * t) + (p0_kick / freqs) * torch.sin(freqs * t)
    p_c = p0_kick * torch.cos(freqs * t) - x_prime_0 * freqs * torch.sin(freqs * t)
    
    x_c = shifts_TE + x_c_prime
    
    exponent = -0.5 * torch.sum(freqs * (walkers - x_c)**2, dim=1)
    
    phase_1 = torch.sum(p_c * (walkers - shifts_TE - x_c_prime / 2), dim=1)
    phase_2 = -0.5 * torch.sum(p0_kick * x_prime_0)
    phase_3 = -0.5 * torch.sum(freqs) * t
    
    log_norm = 0.25 * torch.sum(torch.log(freqs / np.pi))
    
    return torch.complex(log_norm + exponent, phase_1 + phase_2 + phase_3)

def fidelity(net, walkers, t_val, shifts_TE, shifts_GS, p0_kick, freqs):
    with torch.no_grad():
        logpsi = net(walkers).squeeze()
        logphi = exact_evolution_log(walkers, t_val, shifts_TE, shifts_GS, p0_kick, freqs)
        
        log_ratio = logphi - logpsi
        f = torch.exp(log_ratio)
        n_walkers = f.shape[0]
        
        u = f.real
        v = f.imag
        u_mean = torch.mean(u)
        v_mean = torch.mean(v)
        N = u_mean**2 + v_mean**2
        
        D = torch.abs(f)**2
        D_mean = torch.mean(D)
        
        matrix = torch.stack([u, v, D])
        cov_matrix = torch.cov(matrix)
        
        delta_u_sq = cov_matrix[0, 0] / n_walkers
        delta_v_sq = cov_matrix[1, 1] / n_walkers
        delta_D_sq = cov_matrix[2, 2] / n_walkers

        cov_uv = cov_matrix[0, 1] / n_walkers
        cov_uD = cov_matrix[0, 2] / n_walkers
        cov_vD = cov_matrix[1, 2] / n_walkers
        
        delta_N_sq = 4 * (u_mean**2) * delta_u_sq + 4 * (v_mean**2) * delta_v_sq + 8 * u_mean * v_mean * cov_uv
        cov_ND = 2 * u_mean * cov_uD + 2 * v_mean * cov_vD
        
        fidelity_val = N / D_mean
        err_fidelity = fidelity_val * torch.sqrt((delta_N_sq / (N**2)) + (delta_D_sq / (D_mean**2)) - (2 * cov_ND / (N * D_mean)))
        
    return fidelity_val, err_fidelity

def fidelity_grid(net, grid, t_val, shifts_TE, shifts_GS, p0_kick, freqs):
    with torch.no_grad():
        logpsi = net(grid).squeeze()
        logphi = exact_evolution_log(grid, t_val, shifts_TE, shifts_GS, p0_kick, freqs)
        
        psi = torch.exp(logpsi)
        phi = torch.exp(logphi)
        
        overlap = torch.sum(phi.conj() * psi)
        norm_nqs_sq = torch.sum(psi.abs()**2)
        norm_exact_sq = torch.sum(phi.abs()**2)
        
        fidelity_val = (overlap.abs()**2) / (norm_exact_sq * norm_nqs_sq)
        return fidelity_val.real

def run_time_evolution(net, walkers, dt, total_time, n_walkers, freqs_sq, shifts_TE, shifts_GS, device, X_train, kick, step_sizeRK, epsilon):
    time_steps = int(total_time / dt)
    net.kick_momentum.copy_(kick)
    net.apply_kick = True
    
    freqs=torch.sqrt(freqs_sq)

    time_history = torch.zeros(time_steps+1, device=device)
    energy_history = torch.zeros(time_steps+1, device=device)
    pos_history = torch.zeros((time_steps+1, 3), device=device)
    std_history = torch.zeros((time_steps+1, 3), device=device)
    fidelity_history = torch.zeros(time_steps+1, device=device)
    err_fidelity_history = torch.zeros(time_steps+1, device=device)
    fidelity_grid_history = torch.zeros(time_steps+1, device=device)
    std_energy_history = torch.zeros(time_steps+1, device=device)

    # Value of observables at t=0.

    current_time=0

    EL_current, _, _, _ = local_energy(net, walkers, freqs_sq, shifts_TE)

    with torch.no_grad():
        fid, err_fidelity = fidelity(
            net,
            walkers,
            current_time,
            shifts_TE,
            shifts_GS,
            kick,
            freqs
        )

        fid_grid = fidelity_grid(
            net,
            X_train.detach(),
            current_time,
            shifts_TE,
            shifts_GS,
            kick,
            freqs
        )

        E_mean = EL_current.mean().real
        mean_pos = walkers.mean(dim=0)
        std_pos = walkers.std(dim=0)
        std_energy = EL_current.real.std() / math.sqrt(n_walkers)

        time_history[0] = current_time
        energy_history[0] = E_mean
        pos_history[0] = mean_pos
        std_history[0] = std_pos
        fidelity_history[0] = fid
        err_fidelity_history[0] = err_fidelity
        fidelity_grid_history[0] = fid_grid
        std_energy_history[0] = std_energy

    for t in tqdm(range(time_steps), desc="Time Evolution"):
        walkers, EL_current = step_time_rk2(net, walkers, dt, n_walkers, freqs_sq, shifts_TE, step_sizeRK, steps_per_iteration=100, epsilon=epsilon)

        current_time = (t + 1) * dt
        save_idx=t+1
        with torch.no_grad():
            fid, err_fidelity = fidelity(net, walkers, current_time, shifts_TE, shifts_GS, kick, freqs)
            fid_grid = fidelity_grid(net, X_train.detach(), current_time, shifts_TE, shifts_GS, kick, freqs)
            
            E_mean = EL_current.mean().real
            mean_pos = walkers.mean(dim=0)
            std_pos = walkers.std(dim=0)
            std_energy = EL_current.real.std()/math.sqrt(n_walkers)
            
            time_history[save_idx] = current_time
            energy_history[save_idx] = E_mean.real
            pos_history[save_idx] = mean_pos
            std_history[save_idx] = std_pos
            fidelity_history[save_idx] = fid
            err_fidelity_history[save_idx] = err_fidelity
            fidelity_grid_history[save_idx] = fid_grid
            std_energy_history[save_idx] = std_energy

    return {
        't_hist': time_history.cpu().numpy(),
        'e_hist': energy_history.cpu().numpy(),
        'p_hist': pos_history.cpu().numpy(),
        's_hist': std_history.cpu().numpy(),
        'f_hist': fidelity_history.cpu().numpy(),
        'err_f_hist': err_fidelity_history.cpu().numpy(),
        'f_grid_hist': fidelity_grid_history.cpu().numpy(),
        'std_e_hist': std_energy_history.cpu().numpy()
    }