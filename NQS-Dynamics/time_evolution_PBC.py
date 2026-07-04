# time_evolution_deuteron_PBC.py

import torch
import math
import numpy as np
from tqdm import tqdm
from torch.func import functional_call, vmap, grad as func_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ground_state_PBC import local_energy

torch.set_default_dtype(torch.float64)

def tdvp(net, walkers, EL, config):
    n_walkers = config['n_walkers']
    hbar_c = config['hbar_c']
    
    epsilon = config['epsilonTDVP']

    walkers_det = walkers.detach()
    EL_det = EL.detach()
    
    params = dict(net.named_parameters())

    def compute_logpsi_real(p, x):
        return functional_call(net, p, (x.unsqueeze(0),)).squeeze().real

    def compute_logpsi_imag(p, x):
        return functional_call(net, p, (x.unsqueeze(0),)).squeeze().imag

    # Jacobian using vmap (same as SR).
    jac_real_fn = vmap(func_grad(compute_logpsi_real), in_dims=(None, 0))
    jac_imag_fn = vmap(func_grad(compute_logpsi_imag), in_dims=(None, 0))

    jac_real_dict = jac_real_fn(params, walkers_det)
    jac_imag_dict = jac_imag_fn(params, walkers_det)

    flattened_real_grads = []
    flattened_imag_grads = []

    # Loop through every layer's gradient in the dictionary.
    for j_real in jac_real_dict.values():
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
    O_prim = jac - jac_mean

    # Now F = Im(<O* E_L>).
    F = (O_prim.mH @ (EL_det - EL_det.mean())).imag.squeeze(1) / n_walkers
    F = F / hbar_c

    S = (O_prim.mH@O_prim).real/n_walkers
    
    aux = 1+torch.abs(F)
    S = S+epsilon*torch.diag(aux)

    L = torch.linalg.cholesky(S)
    dp = torch.cholesky_solve(F.unsqueeze(1), L).squeeze(1)

    return dp

# RK2 and RK4 functions, although RK2 function is the one used.
def step_time_rk2(net, walkers, dt, config, step_size, steps_per_iteration=500):
    # Save original parameters
    with torch.no_grad():
        p0 = parameters_to_vector(net.parameters())

    # K1
    walkers1, _ = net.metropolis_step(walkers, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL1, _, _, _ = local_energy(net, walkers1, config)
    k1 = tdvp(net, walkers1, EL1, config)

    with torch.no_grad():
        p_temp = p0 + dt * k1
        vector_to_parameters(p_temp, net.parameters())

    # K2.
    walkers2, _ = net.metropolis_step(walkers1, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL2, _, _, _ = local_energy(net, walkers2, config)
    k2 = tdvp(net, walkers2, EL2, config)

    # Combine.
    with torch.no_grad():
        p_new = p0 + 0.5 * dt * (k1 + k2)
        vector_to_parameters(p_new, net.parameters())
    
    walkers_final, _ = net.metropolis_step(
        walkers2, step_size, N_acc=0, steps_per_iteration=steps_per_iteration)
    EL_final, _, _, _ = local_energy(net, walkers_final, config)

    return walkers_final.detach(), EL_final.detach()



def step_time_rk4(net, walkers, dt, config, step_size, steps_per_iteration=500):
    
    params_list = list(net.parameters())

    with torch.no_grad():
        p0 = parameters_to_vector(params_list).detach().clone()

    # K1
    walkers1, _ = net.metropolis_step(
        walkers, step_size, N_acc=0, steps_per_iteration=steps_per_iteration
    )
    EL1, _, _, _ = local_energy(net, walkers1, config)
    k1 = tdvp(net, walkers1, EL1, config).detach()

    # K2
    with torch.no_grad():
        vector_to_parameters(p0 + 0.5 * dt * k1, params_list)

    walkers2, _ = net.metropolis_step(
        walkers1, step_size, N_acc=0, steps_per_iteration=steps_per_iteration
    )
    EL2, _, _, _ = local_energy(net, walkers2, config)
    k2 = tdvp(net, walkers2, EL2, config).detach()

    # K3
    with torch.no_grad():
        vector_to_parameters(p0 + 0.5 * dt * k2, params_list)

    walkers3, _ = net.metropolis_step(
        walkers2, step_size, N_acc=0, steps_per_iteration=steps_per_iteration
    )
    EL3, _, _, _ = local_energy(net, walkers3, config)
    k3 = tdvp(net, walkers3, EL3, config).detach()

    # K4
    with torch.no_grad():
        vector_to_parameters(p0 + dt * k3, params_list)

    walkers4, _ = net.metropolis_step(
        walkers3, step_size, N_acc=0, steps_per_iteration=steps_per_iteration
    )
    EL4, _, _, _ = local_energy(net, walkers4, config)
    k4 = tdvp(net, walkers4, EL4, config).detach()

    # Combine.
    with torch.no_grad():
        p_new = p0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        vector_to_parameters(p_new, params_list)

    walkers_final, _ = net.metropolis_step(
        walkers4, step_size, N_acc=0, steps_per_iteration=steps_per_iteration
    )
    EL_final, _, _, _ = local_energy(net, walkers_final, config)

    return walkers_final.detach(), EL_final.detach()


def run_time_evolution(net, walkers, config, exact_data_te, X_train):
    device = config['device']
    dt = config['dt']
    total_time = config['total_time']
    time_steps = int(total_time / dt)
    save_interval = config['save_interval']
    num_saves = time_steps // save_interval + 1
    step_size = config['step_sizeRK']

    dV = config['dV']
    n_walkers = config['n_walkers']
    center_potential = config['center_potential']
    gamma = config['gamma_boost']
    
    if exact_data_te is not None:
        times_te = exact_data_te['times']
        r_te = exact_data_te['r']
        u_te = exact_data_te['u_history']
    else:
        times_te, r_te, u_te = None, None, None

    history = {
        'time': torch.zeros(num_saves, device=device),
        'energy': torch.zeros(num_saves, device=device),
        'pos': torch.zeros((num_saves, 3), device=device),
        'std': torch.zeros((num_saves, 3), device=device),
        'overlap': torch.zeros(num_saves, device=device),
        'r_mean': torch.zeros(num_saves, device=device),
        'r_mean_walkers': torch.zeros(num_saves, device=device),
        'r_mean_walkers_err': torch.zeros(num_saves, device=device),
        'core_prob': torch.zeros(num_saves, device=device),
        'fidelity': torch.zeros(num_saves, device=device),
        'fidelity_mc': torch.zeros(num_saves, device=device),
        'energy_err': torch.zeros(num_saves, device=device),
        'fidelity_mc_err': torch.zeros(num_saves, device=device)
    }

    with torch.no_grad():
        chunk_size = 5000000
        logpsi_list = []

        for start_idx in range(0, X_train.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, X_train.shape[0])
            logpsi_list.append(net(X_train[start_idx:end_idx]))

        logpsi_initial = torch.cat(logpsi_list, dim=0).squeeze()
        psi_0 = torch.exp(logpsi_initial)

        r2_mesh = torch.sum((X_train - center_potential)**2, dim=1)
        r_mesh = torch.sqrt(r2_mesh)

        r_mesh_np = r_mesh.cpu().numpy() 

        norm_0_sq = torch.sum(psi_0.abs()**2) * dV

    save_idx = 0
    
    # There was a final SR update.
    walkers = net.thermalization(walkers.detach(), config['step_sizeRK'])
    walkers, _ = net.metropolis_step(walkers,config['step_sizeRK'],N_acc=0,steps_per_iteration=500)

    # Observables at t=0.
    EL0, _, _, _ = local_energy(net, walkers, config)
    with torch.no_grad():
        E_mean = EL0.mean().real
        mean_pos = walkers.mean(dim=0)
        std_pos = walkers.std(dim=0)

        # At t = 0, psi_t is psi_0.
        psi_t = psi_0
        prob_density_3d = psi_t.abs()**2
        norm_t_sq = norm_0_sq

        overlap_val = torch.tensor(1.0, dtype=torch.float64, device=device)

        r_exp_val = torch.sum(r_mesh * prob_density_3d) * dV / norm_t_sq  # With mesh integral.
        r_walkers=torch.sqrt(
                    torch.sum((walkers-center_potential)**2, dim=1))
        r_exp_val_walkers=r_walkers.mean()
        r_exp_val_walkers_err=r_walkers.std()/math.sqrt(n_walkers)

        core_mask = r_mesh < 1
        core_prob_val = torch.sum(prob_density_3d[core_mask]) * dV / norm_t_sq

        # Exact numerical state at t = 0.
        u_exact_t = u_te[0]
        psi_exact_1d_t = u_exact_t / ((r_te + 1e-10) * np.sqrt(4 * math.pi))

        psi_real_np = np.interp(r_mesh_np, r_te, psi_exact_1d_t.real)
        psi_imag_np = np.interp(r_mesh_np, r_te, psi_exact_1d_t.imag)

        psi_exact_3d = torch.tensor(
            psi_real_np + 1j * psi_imag_np,
            dtype=torch.complex128,
            device=device
        )

        norm_exact_sq = torch.sum(psi_exact_3d.abs()**2) * dV

        fidelity_int = torch.sum(psi_exact_3d.conj() * psi_t) * dV
        fidelity_val = (fidelity_int.abs()**2) / (norm_exact_sq * norm_t_sq)

        std_energy = EL0.std().real / math.sqrt(n_walkers)

        history['time'][save_idx] = 0.0
        history['energy'][save_idx] = E_mean
        history['pos'][save_idx] = mean_pos
        history['std'][save_idx] = std_pos
        history['overlap'][save_idx] = overlap_val
        history['r_mean'][save_idx] = r_exp_val
        history['r_mean_walkers'][save_idx] = r_exp_val_walkers
        history['r_mean_walkers_err'][save_idx] = r_exp_val_walkers_err
        history['core_prob'][save_idx] = core_prob_val
        history['fidelity'][save_idx] = fidelity_val.real
        history['energy_err'][save_idx] = std_energy

        history['fidelity_mc'][save_idx] = fidelity_val.real
        history['fidelity_mc_err'][save_idx] = 0.0

    for t in tqdm(range(1,time_steps+1), desc="Time Evolution"):

        walkers, EL = step_time_rk4(
            net, walkers, dt, config, step_size, steps_per_iteration=500)

        if t % save_interval == 0:
            save_idx += 1
            current_time = t*dt

            with torch.no_grad():
                E_mean = EL.mean().real
                mean_pos = walkers.mean(dim=0)
                std_pos = walkers.std(dim=0)

                # Current evolved wavefunction.
                logpsi_list = []
                for start_idx in range(0, X_train.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, X_train.shape[0])
                    logpsi_list.append(net(X_train[start_idx:end_idx]))

                logpsi_t = torch.cat(logpsi_list, dim=0).squeeze()
                psi_t = torch.exp(logpsi_t)

                prob_density_3d = psi_t.abs()**2
                norm_t_sq = torch.sum(prob_density_3d) * dV

                # Overlap between initial state and evolved state.
                overlap_int = torch.sum(psi_0.conj() * psi_t) * dV
                overlap_val = (overlap_int.abs()**2) / (norm_0_sq * norm_t_sq)
                
                # Radius <r>.               
                r_exp_val = torch.sum(r_mesh * prob_density_3d) * dV / norm_t_sq  # With mesh integral.
                r_walkers=torch.sqrt(
                    torch.sum((walkers-center_potential)**2, dim=1))
                r_exp_val_walkers=r_walkers.mean()
                r_exp_val_walkers_err=r_walkers.std()/math.sqrt(n_walkers)


                # Core probability (r<1 fm).
                core_mask = r_mesh < 1
                core_prob_val = torch.sum(
                    prob_density_3d[core_mask]) * dV / norm_t_sq

                # Fidelity between NQS and numerical state.

                # 1D numerical state (u(r), not psi(r)).
                u_exact_t = u_te[save_idx]

                # Convert numerical u(r) into 3D psi(r). 1/sqrt(4*pi) because of spherical harmonic Y^0_0.
                psi_exact_1d_t = u_exact_t / \
                    ((r_te + 1e-10) * np.sqrt(4 * math.pi))

                # Interpolate real and imaginary parts to have the same number of points as the grid.
                psi_real_np = np.interp(r_mesh_np, r_te, psi_exact_1d_t.real)
                psi_imag_np = np.interp(r_mesh_np, r_te, psi_exact_1d_t.imag)

                psi_exact_3d = torch.tensor(
                    psi_real_np + 1j * psi_imag_np, dtype=torch.complex128, device=device)

                norm_exact_sq = torch.sum(psi_exact_3d.abs()**2) * dV

                # Fidelity with grid. |<psi_exact|psi_NQS>|^2
                fidelity_int = torch.sum(psi_exact_3d.conj() * psi_t) * dV
                fidelity_val = (fidelity_int.abs()**2) / \
                    (norm_exact_sq * norm_t_sq)

                # Fidelity with walkers.
                r_walkers_np = r_walkers.cpu().numpy()

                # Interpolate the exact state onto the walkers' continuous positions.
                psi_exact_w_real = np.interp(
                    r_walkers_np, r_te, psi_exact_1d_t.real)
                psi_exact_w_imag = np.interp(
                    r_walkers_np, r_te, psi_exact_1d_t.imag)
                psi_exact_w = torch.tensor(
                    psi_exact_w_real + 1j * psi_exact_w_imag, dtype=torch.complex128, device=device)

                # Evaluate the NQS at the walker positions.
                psi_nqs_w = torch.exp(net(walkers).squeeze())

                # Compute f(r)=psi_exact/psi_NQS.
                f_ratio = psi_exact_w / (psi_nqs_w + 1e-10)

                # |<f>|^2/<|f|^2>
                f_mean = torch.mean(f_ratio)
                f_sq_mean = torch.mean(f_ratio.abs()**2)
                fidelity_mc_val = (f_mean.abs()**2) / f_sq_mean

                # Errors.

                # Energy statistical error.
                std_energy = EL.std().real / math.sqrt(n_walkers)

                # Fidelity walkers error.
                u_vals = f_ratio.real
                v_vals = f_ratio.imag
                D_vals = f_ratio.abs()**2

                u_mean = u_vals.mean()
                v_mean = v_vals.mean()
                D_mean = D_vals.mean()
                N_mean = u_mean**2 + v_mean**2

                Delta_u = u_vals.std() / math.sqrt(n_walkers)
                Delta_v = v_vals.std() / math.sqrt(n_walkers)
                Delta_D = D_vals.std() / math.sqrt(n_walkers)

                data_matrix = torch.stack([u_vals, v_vals, D_vals], dim=0)
                cov_matrix = torch.cov(data_matrix) / n_walkers

                cov_uv = cov_matrix[0, 1]
                cov_uD = cov_matrix[0, 2]
                cov_vD = cov_matrix[1, 2]

                term1 = (4 * u_mean**2 * Delta_u**2 + 4 * v_mean**2 * Delta_v **
                         2 + 8 * u_mean * v_mean * cov_uv) / (N_mean**2 + 1e-10)
                term2 = (Delta_D / D_mean)**2
                term3 = 4 * (u_mean * cov_uD + v_mean * cov_vD) / \
                    (N_mean * D_mean + 1e-10)

                delta_F = fidelity_mc_val * torch.sqrt(term1 + term2 - term3)

                history['time'][save_idx] = current_time
                history['energy'][save_idx] = E_mean
                history['pos'][save_idx] = mean_pos
                history['std'][save_idx] = std_pos
                history['overlap'][save_idx] = overlap_val
                history['r_mean'][save_idx] = r_exp_val
                history['r_mean_walkers'][save_idx] = r_exp_val_walkers
                history['r_mean_walkers_err'][save_idx] = r_exp_val_walkers_err
                history['core_prob'][save_idx] = core_prob_val
                history['fidelity'][save_idx] = fidelity_val.real
                history['energy_err'][save_idx] = std_energy
                history['fidelity_mc_err'][save_idx] = delta_F
                history['fidelity_mc'][save_idx] = fidelity_mc_val.real

    return history
