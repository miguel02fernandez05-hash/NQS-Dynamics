# main_deuteron.py

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from ground_state import train_ground_state, DeuteronNQS, local_energy
import os
from time_evolution import run_time_evolution

import matplotlib as mpl

# Format for figures.
mpl.rcParams.update({
    "text.usetex": False, 
    "font.family": "serif",
    "mathtext.fontset": "cm",  
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

torch.set_default_dtype(torch.float64)

# Current values are for the Heavy Deuteron.
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    m_p, m_n = 938.272088, 939.565420
    mass_factor=10
    m_p_eff=mass_factor*m_p
    m_n_eff=mass_factor*m_n
    mu = (m_p_eff*m_n_eff) / (m_p_eff+m_n_eff)
    hbar_c = 197.32698

    config = {
        'device': device,
        'hbar_c': hbar_c,
        'm_p': m_p,
        'm_n': m_n,
        'mu': mu,
        'mass_factor': mass_factor,
        'kin_prefactor': (hbar_c**2) / (2*mu),
        'C_10': -142,
        'R_0': 1,
        'x0': 0, 'y0': 0, 'z0': 0,
        'center_potential': torch.tensor([0, 0, 0], dtype=torch.float64, device=device),
        'Nin': 3, 'Nout': 1, 'Nhid': 20,
        'seed': 1,
        'pretraining_epochs': 0,
        'epochs': 800,
        'initial_lr': 0.003,
        'final_lr': 0.003,
        'step_size_gs': 0.5,
        'steps_per_iteration_gs': 100,
        'epsilonSR': 4e-5,
        'n_walkers': 50000,
        'Nx': 300,
        'train_a': -3.5, 'train_b': 3.5,
        'gamma_boost':0.1,
        'step_sizeRK':0.5,
        'epsilonTDVP':5e-7,
        'dt': 0.05,
        'total_time': 26,
        'save_interval': 1
    }
    config['h'] = (config['train_b']-config['train_a'])/(config['Nx']-1)
    config['dV'] = config['h']**3

    # Mesh for grid calculations.
    coords = torch.linspace(config['train_a'], config['train_b'], config['Nx'])
    x_mesh, y_mesh, z_mesh = torch.meshgrid(coords, coords, coords, indexing='ij')
    inputs_flat = torch.stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()], dim=1)
    X_train = inputs_flat.to(device).requires_grad_(True)
    coords_plot = coords.numpy()

    base_filename = (f"Factor_{config['mass_factor']}_"
                f"Nhid_{config['Nhid']}_"
                f"C10_{config['C_10']}_"
                f"Walkers_{config['n_walkers']}_"
                f"InitialLR_{config['initial_lr']}_"
                f"Step_size_gs_{config['step_size_gs']}_"
                f"epsilonSR_{config['epsilonSR']}_"
                f"dt_{config['dt']}_"
                f"gamma_{config['gamma_boost']}_"
                f"epsilonTDVP_{config['epsilonTDVP']}_"
                f"step_sizeRK_{config['step_sizeRK']}_"
                f"Save_interval_{config['save_interval']}_")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Folder where the exact numerical results .npz files are.
    exact_folder = os.path.join(script_dir, "Exact")
    # Folder where the plots are saved.
    run_folder = os.path.join(script_dir, "Plots_Mass_Factor_10", base_filename)
    os.makedirs(run_folder, exist_ok=True)

    # Load exact numerical results.
    exact_data_gs = np.load(os.path.join(
        exact_folder, "exact_deuteron_target_C_10_142_Mass_Factor_10.npz"))
    r_exact_gs, Psi_exact = exact_data_gs['r'], exact_data_gs['psi']
    r_exact_target, r2_exact_target = exact_data_gs['r_expected_exact'].item(
    ), exact_data_gs['r2_expected_exact'].item()
    E_exact_target = exact_data_gs['E'].item()

    exact_data_te = np.load(os.path.join(
        exact_folder, "exact_time_evolution_0.1_Mass_Factor_10.npz"))

    # Initial value of the weight matrices and bias vectors.
    torch.manual_seed(config['seed'])
    W1_amp = torch.rand(config['Nhid'], config['Nin'],
                        requires_grad=True)*(-1.)
    B1_amp = torch.rand(config['Nhid'], requires_grad=True)*2.-1.
    W2_amp = torch.rand(config['Nhid'], config['Nhid'],
                        requires_grad=True)*(-1.)
    B2_amp = torch.rand(config['Nhid'], requires_grad=True)*2.-1.
    W3_amp = torch.rand(config['Nout'], config['Nhid'], requires_grad=True)

    W1_phase = torch.rand(
        config['Nhid'], config['Nin'], requires_grad=True)*(-1.)
    B1_phase = torch.rand(config['Nhid'], requires_grad=True)*2.-1.
    W2_phase = torch.rand(
        config['Nhid'], config['Nhid'], requires_grad=True)*(-1.)
    B2_phase = torch.rand(config['Nhid'], requires_grad=True)*2.-1.
    W3_phase = torch.rand(config['Nout'], config['Nhid'], requires_grad=True)

    net = DeuteronNQS(W1_amp, B1_amp, W2_amp, B2_amp, W3_amp,
                      W1_phase, B1_phase, W2_phase, B2_phase, W3_phase,
                      config['Nin'], config['Nhid'], config['Nout'],
                      config['x0'], config['y0'], config['z0']).to(device)

    # This would be used for pretraining on a grid, but pretraining_epochs is set to 0 because it's not needed.
    optimizer = torch.optim.RMSprop(params=net.parameters(), lr=config['initial_lr'])
    # optimizer=torch.optim.Adam(params=net.parameters(), lr=lr)



    # This callback was originally used for local interactive runs, where the
    # training plot could be refreshed while the ground-state optimization was
    # still running. On Artemisa, there is no interactive display, so the live
    # refresh is disabled by keeping plt.pause(0.01) commented out inside the
    # callback. In order to bring back the interactive feature that updates the
    # figure while the code is running, uncomment the line plt.pause(0.01) and
    # comment the line matplotlib.use('Agg') at the imports on the top of the script.


    plt.ion() 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5.8, 5.2))
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Function to update figure while code is still running.
    def pic_callback(i, Psi_mesh, Psi_norm_mesh, loss_accum, U_accum, K_accum):
        Psi_norm_reshaped = np.abs(Psi_norm_mesh.detach().cpu().numpy()).reshape(
            config['Nx'], config['Nx'], config['Nx'])
        idx_x0 = (np.abs(coords_plot-config['x0'])).argmin()
        idx_y0 = (np.abs(coords_plot-config['y0'])).argmin()
        idx_z0 = (np.abs(coords_plot-config['z0'])).argmin()

        slice_x_nqs = Psi_norm_reshaped[:, idx_y0, idx_z0]
        slice_y_nqs = Psi_norm_reshaped[idx_x0, :, idx_z0]
        slice_z_nqs = Psi_norm_reshaped[idx_x0, idx_y0, :]
        target_slice = np.interp(np.abs(coords_plot), r_exact_gs, Psi_exact)

        def update_plot(ax, x_data, y_nqs, y_tar, title):
            max_val = max(np.max(y_nqs), np.max(y_tar))
            ax.set_ylim(-0.05, max_val*1.2 if max_val > 0 else 1)
            ax.set_title(title)
            if ax.lines:
                ax.lines[0].set_data(x_data, y_nqs)
                ax.lines[1].set_data(x_data, y_tar)
            else:
                ax.plot(x_data, y_nqs, label=r'$\Psi_{NQS}$', color='b', linewidth=1.2)
                ax.plot(x_data, y_tar, '--', label=r'$\Psi_{Target}$', color='r', linewidth=1.2)
                ax.legend(loc='upper right', fontsize='small')

        update_plot(ax1, coords_plot, slice_x_nqs, target_slice, r"Wavefunction on X-axis")
        update_plot(ax2, coords_plot, slice_y_nqs, target_slice, r"Wavefunction on Y-axis")
        update_plot(ax3, coords_plot, slice_z_nqs, target_slice, r"Wavefunction on Z-axis")

        x_epochs = np.arange(1, i+2)
        y_loss = loss_accum[:i+1].detach().cpu().numpy()
        y_U = U_accum[:i+1].detach().cpu().numpy()
        y_K = K_accum[:i+1].detach().cpu().numpy()

        ax4.cla()
        ax4.set_xlim(0, i+1)
        # ax1.set_xlim(-10.5, 10.5)
        # ax2.set_xlim(-10.5, 10.5)
        # ax3.set_xlim(-10.5, 10.5)
        # ax1.set_ylim(-0.025, 0.29)
        # ax2.set_ylim(-0.025, 0.29)
        # ax3.set_ylim(-0.025, 0.29)
        
        ax1.set_ylabel(r"$\Psi(x,0,0)$")
        ax2.set_ylabel(r"$\Psi(0,y,0)$")
        ax3.set_ylabel(r"$\Psi(0,0,z)$")

        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
        ax3.grid(alpha=0.3)
        ax4.grid(alpha=0.3)

        # ax4.set_ylim(-2.5, 0)
        ax4.set_xlabel(r"Epoch")
        ax4.set_ylabel(r"Energy (MeV)")
        ax4.set_title(r"Energy")
        ax4.plot(x_epochs, y_loss, label=r'$E$', color='b', linewidth=1.2)
        ax4.plot(x_epochs, y_U, label=r'$U$', color='orange', linewidth=0.8, alpha=0.8)
        ax4.plot(x_epochs, y_K, label=r'$K$', color='green', linewidth=0.8, alpha=0.8)
        
        ax4.plot(x_epochs, np.ones_like(x_epochs)*(E_exact_target), '--',label=rf'$E_0 \approx {E_exact_target:.2f}$ MeV', color='k', alpha=0.6)
        
        ax4.legend(loc='upper right', fontsize='small')
        fig.canvas.draw()
        fig.canvas.flush_events()

        # plt.pause(0.01)


    print("NN architecture:\n", net)

    # Ground state optimization.

    net, walkers, gs_hist = train_ground_state(
        net, optimizer, config, X_train, plot_callback=pic_callback)
    plt.ioff()

    training_plot_path_png = os.path.join(run_folder, "0_training.png")
    training_plot_path_pdf = os.path.join(run_folder, "training.pdf")

    fig.savefig(training_plot_path_png, bbox_inches='tight', dpi=300)
    fig.savefig(training_plot_path_pdf, bbox_inches='tight')
    plt.close(fig)

    aux = config['pretraining_epochs'] + config['epochs']
    mesh_epochs_np = gs_hist['mesh_epochs'].cpu().numpy()
    r_mesh_accum_np = gs_hist['r_mesh_accum'].cpu().numpy()
    r2_mesh_accum_np = gs_hist['r2_mesh_accum'].cpu().numpy()
    r_walkers_accum_np = gs_hist['r_walkers_accum'].cpu().numpy()
    r2_walkers_accum_np = gs_hist['r2_walkers_accum'].cpu().numpy()
    r_walkers_err_np = gs_hist['r_walkers_err_accum'].cpu().numpy()
    r2_walkers_err_np = gs_hist['r2_walkers_err_accum'].cpu().numpy()
    r_mesh_err_np = gs_hist['r_mesh_err_accum'].cpu().numpy()
    r2_mesh_err_np = gs_hist['r2_mesh_err_accum'].cpu().numpy()
    loss_accum = gs_hist['loss_accum']
    U_accum = gs_hist['U_accum']
    K_accum = gs_hist['K_accum']
    norm_const = gs_hist['norm_const']
    acceptance = gs_hist['acceptance']

    # Plot 1: radial wavefunction.
    plt.figure(figsize=(8, 5), dpi=300)
    r_nqs = torch.linspace(0.01, 5, 500, device=device)
    eval_points = torch.stack(
        [r_nqs, torch.zeros_like(r_nqs), torch.zeros_like(r_nqs)], dim=1)

    with torch.no_grad():
        logPsi_nqs = net(eval_points)
        Psi_nqs = torch.exp(logPsi_nqs).abs().squeeze().cpu().numpy()

    Psi_nqs_normalized = Psi_nqs / norm_const.item()
    r_nqs_np = r_nqs.cpu().numpy()

    safe_idx = r_exact_gs > 0.01
    r_exact_safe = r_exact_gs[safe_idx]
    Psi_exact_safe = Psi_exact[safe_idx]

    plt.plot(r_nqs_np, Psi_nqs_normalized, 'b',label='NQS $\Psi(r)$', linewidth=1.5)
    plt.plot(r_exact_safe, Psi_exact_safe, 'r--',label='Numerical $\Psi(r)$', alpha=0.8, linewidth=1.5)
    plt.xlim(0, 3.5)
    plt.xlabel("Radius $r$ (fm)")
    plt.ylabel("Wavefunction $\Psi(r)$")
    plt.title("$\Psi(r)$")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        run_folder, f"1_Psi_r.png"), dpi=300)
    plt.close()

    # Plot 2 and 3: walkers' histogram.
    walkers_np = walkers.detach().cpu().numpy()
    plt.figure(figsize=(8, 5), dpi=300)
    plt.grid(alpha=0.3)
    plt.hist(walkers_np[:, 0], bins=80, density=True, label='Walkers X')
    plt.legend()
    plt.savefig(os.path.join(
        run_folder, f"2_walkers_initial.png"), dpi=300)
    plt.close()
    x_coords = walkers_np[:, 0]
    y_coords = walkers_np[:, 1]
    z_coords = walkers_np[:, 2]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(x_coords, bins=80, density=True,alpha=0.6, color='C0', label='Walkers X')
    plt.hist(y_coords, bins=80, density=True,alpha=0.6, color='C1', label='Walkers Y')
    plt.hist(z_coords, bins=80, density=True,alpha=0.6, color='C2', label='Walkers Z')
    plt.title('Sampled densities')
    plt.xlabel('Position')
    plt.ylabel('$|\Psi(x,y,z)|^2$')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(run_folder, f"3_walkers_final.png"), dpi=300)
    plt.close()

    # Plot 4: <r>.
    plt.figure(figsize=(8, 5), dpi=300)
    epochs_arr = np.arange(aux)
    start_idx = config['pretraining_epochs']

    plt.plot(epochs_arr[start_idx:], r_walkers_accum_np[start_idx:],
            color='green', label=f'Walkers $\\langle r \\rangle$', linewidth=1.5)
    plt.fill_between(epochs_arr[start_idx:],
                    r_walkers_accum_np[start_idx:] -
                    r_walkers_err_np[start_idx:],
                    r_walkers_accum_np[start_idx:] +
                    r_walkers_err_np[start_idx:],
                    color='green', alpha=0.3)

    plt.plot(mesh_epochs_np, r_mesh_accum_np, 'bo-',
            label=r'NQS Mesh Integral $\langle r \rangle$', markersize=5, linewidth=1.5)
    plt.fill_between(mesh_epochs_np,
                    r_mesh_accum_np - r_mesh_err_np,
                    r_mesh_accum_np + r_mesh_err_np,
                    color='blue', alpha=0.3)

    plt.axhline(y=r_exact_target, color='r', linestyle='--',
                label=rf'Numerical $\langle r \rangle \approx$ {r_exact_target:.3f} fm')
    plt.xlim(0, aux)
    plt.ylim(0, max(np.max(r_mesh_accum_np)*1.2, r_exact_target*1.5))
    plt.xlabel("Epoch")
    plt.ylabel("$\langle r \\rangle$ (fm)")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_folder, f"4_r.png"), dpi=300)
    plt.close()

    # Plot 5: sqrt(<r^2>)/2.
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs_arr[start_idx:], r2_walkers_accum_np[start_idx:],
            color='green', label=r'Walkers $\frac{\sqrt{\langle r^2 \rangle}}{2}$', linewidth=1.5)
    plt.fill_between(epochs_arr[start_idx:],
                    r2_walkers_accum_np[start_idx:] -
                    r2_walkers_err_np[start_idx:],
                    r2_walkers_accum_np[start_idx:] +
                    r2_walkers_err_np[start_idx:],
                    color='green', alpha=0.3)

    plt.plot(mesh_epochs_np, r2_mesh_accum_np, 'bo-',
             label=r'NQS Mesh Integral $\frac{\sqrt{\langle r^2 \rangle}}{2}$', markersize=5, linewidth=1.5)
    plt.fill_between(mesh_epochs_np,
                     r2_mesh_accum_np - r2_mesh_err_np,
                     r2_mesh_accum_np + r2_mesh_err_np,
                     color='blue', alpha=0.3)

    plt.axhline(y=r2_exact_target, color='r', linestyle='--',
                label=rf'Numerical $\frac{{\sqrt{{\langle r^2 \rangle}}}}{{2}} \approx$ {r2_exact_target:.3f} fm')
    plt.xlim(0, aux)
    plt.ylim(0, max(np.max(r2_mesh_accum_np)*1.2, r2_exact_target*1.5))
    plt.xlabel("Epoch")
    plt.ylabel(r"$\frac{\sqrt{\langle r^2 \rangle}}{2}$ (fm)")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(
        run_folder, f"5_r_sq.png"), dpi=300)
    plt.close()
    
    # Plot 6: zoomed <r>.
    plt.figure(figsize=(4.6, 3.4), dpi=300)
    plt.plot(epochs_arr[start_idx:], r_walkers_accum_np[start_idx:],
            color='green', label=r'Walkers $\langle r \rangle$', linewidth=1.2)
    plt.fill_between(epochs_arr[start_idx:],
                    r_walkers_accum_np[start_idx:] - r_walkers_err_np[start_idx:],
                    r_walkers_accum_np[start_idx:] + r_walkers_err_np[start_idx:],
                    color='green', alpha=0.3)

    plt.plot(mesh_epochs_np, r_mesh_accum_np, 'bo-',
            label=r'NQS Mesh Integral $\langle r \rangle$', markersize=4, linewidth=1.2)
    plt.fill_between(mesh_epochs_np,
                    r_mesh_accum_np - r_mesh_err_np,
                    r_mesh_accum_np + r_mesh_err_np,
                    color='blue', alpha=0.3)

    plt.axhline(y=r_exact_target, color='r', linestyle='--',
                label=rf'Numerical $\langle r \rangle \approx {r_exact_target:.3f}$ fm', linewidth=1.2)
    plt.xlim(0, aux)
    plt.ylim(0.95*r_exact_target, 1.18*r_exact_target)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"$\langle r \rangle$ (fm)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "6_r_zoomed.png"), dpi=300)
    plt.savefig(os.path.join(run_folder, "r_zoomed.pdf"), bbox_inches='tight')
    plt.close()

    # Plot 7: zoomed sqrt(<r^2>)/2.
    plt.figure(figsize=(4.6, 3.4), dpi=300)
    plt.plot(epochs_arr[start_idx:], r2_walkers_accum_np[start_idx:],
            color='green', label=r'Walkers $\frac{\sqrt{\langle r^2 \rangle}}{2}$', linewidth=1.2)
    plt.fill_between(epochs_arr[start_idx:],
                    r2_walkers_accum_np[start_idx:] - r2_walkers_err_np[start_idx:],
                    r2_walkers_accum_np[start_idx:] + r2_walkers_err_np[start_idx:],
                    color='green', alpha=0.3)

    plt.plot(mesh_epochs_np, r2_mesh_accum_np, 'bo-',
            label=r'NQS Mesh Integral $\frac{\sqrt{\langle r^2 \rangle}}{2}$', markersize=4, linewidth=1.2)
    plt.fill_between(mesh_epochs_np,
                    r2_mesh_accum_np - r2_mesh_err_np,
                    r2_mesh_accum_np + r2_mesh_err_np,
                    color='blue', alpha=0.3)

    plt.axhline(y=r2_exact_target, color='r', linestyle='--',
                label=rf'Numerical $\frac{{\sqrt{{\langle r^2 \rangle}}}}{{2}} \approx {r2_exact_target:.3f}$ fm', linewidth=1.2)
    plt.xlim(0, aux)
    plt.ylim(0.95*r2_exact_target, 1.18*r2_exact_target)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"$\frac{\sqrt{\langle r^2 \rangle}}{2}$ (fm)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "7_r_sq_zoomed.png"), dpi=300)
    plt.savefig(os.path.join(run_folder, "r2_zoomed.pdf"), bbox_inches='tight')
    plt.close()

    # Plot 8: energy zoomed out.
    plt.figure(figsize=(4.6, 3.4), dpi=300)
    x_epochs = np.linspace(1, config['epochs'], config['epochs'])
    y_loss = loss_accum.detach().cpu().numpy()[start_idx:]
    y_U = U_accum.detach().cpu().numpy()[start_idx:]
    y_K = K_accum.detach().cpu().numpy()[start_idx:]

    plt.plot(x_epochs, y_loss, label=r'$E$', color='b', linewidth=1.2)
    plt.plot(x_epochs, y_U, label=r'$U$', color='orange', linewidth=1.0)
    plt.plot(x_epochs, y_K, label=r'$K$', color='green', linewidth=1.0)
    plt.plot(x_epochs, np.ones_like(x_epochs)*(E_exact_target), '--',
            label=rf'$E_0 \approx {E_exact_target:.2f}$ MeV', color='k', alpha=0.6, linewidth=1.2)

    plt.xlim(0, config['epochs'])
    plt.ylim(np.min([np.min(y_loss), np.min(y_U), np.min(y_K)])-1,
            np.max([np.max(y_loss), np.max(y_U), np.max(y_K)])+1)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Energy (MeV)")
    plt.title(r"Energy")
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "8_energy_zoomed_out.png"), dpi=300)
    plt.savefig(os.path.join(run_folder, "energy_zoomed_out.pdf"), bbox_inches='tight') # <-- Added bbox trim!
    plt.close()

    # Energy for prints.
    EL, UL, KL, logPsi_walkers = local_energy(net, walkers, config)

    with torch.no_grad():
        n_samples = EL.shape[0]

        E_mean = EL.real.mean()
        U_mean = UL.real.mean()
        K_mean = KL.real.mean()

        E_std = EL.real.std(unbiased=True)
        U_std = UL.real.std(unbiased=True)
        K_std = KL.real.std(unbiased=True)

        E_err = E_std / math.sqrt(n_samples)
        U_err = U_std / math.sqrt(n_samples)
        K_err = K_std / math.sqrt(n_samples)

        E_var = EL.real.var(unbiased=True)

    print('Acceptance =', acceptance)
    print(
        f"Walkers <r>: {r_walkers_accum_np[-1]:.4f} ± {r_walkers_err_np[-1]:.4f}")
    print(
        f"NQS Mesh Integral <r>: {r_mesh_accum_np[-1]:.4f} ± {r_mesh_err_np[-1]:.4e}")
    print(f"Numerical <r>: {r_exact_target:.4f}")
    print(
        f"Walkers sqrt(<r^2>): {r2_walkers_accum_np[-1]:.4f} ± {r2_walkers_err_np[-1]:.4f}")
    print(
        f"NQS Mesh Integral sqrt(<r^2>): {r2_mesh_accum_np[-1]:.4f} ± {r2_mesh_err_np[-1]:.4e}")
    print(f"Numerical sqrt(<r^2>): {r2_exact_target:.4f}")

    print(f"Energy: {E_mean:.4f} ± {E_err:.4f}")
    print(f"Potential Energy: {U_mean:.4f} ± {U_err:.4f}")
    print(f"Kinetic Energy: {K_mean:.4f} ± {K_err:.4f}")
    print(f"Energy Standard Deviation: {E_std:.4f}")
    print(f"Energy Variance: {E_var:.4f}")

    # Time evolution.

    # Introduce the kick.
    with torch.no_grad():
        net.gamma.fill_(config['gamma_boost'])

    print(f"Starting Time Evolution with gamma={config['gamma_boost']}")

    te_hist=run_time_evolution(net, walkers, config, exact_data_te, X_train)

    time_history=te_hist['time'].cpu().numpy()
    energy_history=te_hist['energy'].cpu().numpy()
    overlap_history=te_hist['overlap'].cpu().numpy()
    r_mean_history=te_hist['r_mean'].cpu().numpy()
    r_mean_walkers_history=te_hist['r_mean_walkers'].cpu().numpy()
    r_mean_walkers_err_history=te_hist['r_mean_walkers_err'].cpu().numpy()
    core_prob_history=te_hist['core_prob'].cpu().numpy()
    fidelity_history=te_hist['fidelity'].cpu().numpy()
    fidelity_mc_history=te_hist['fidelity_mc'].cpu().numpy()
    energy_err_np=te_hist['energy_err'].cpu().numpy()
    fidelity_mc_err_np=te_hist['fidelity_mc_err'].cpu().numpy()

    times_te = exact_data_te['times']
    radius_te = exact_data_te['r_expected']
    core_prob_te = exact_data_te['core_prob']
    survival_prob_te = exact_data_te['survival_prob']
    
    # Linear regression in energy plot.
    coefficients=np.polyfit(time_history, energy_history, 1)
    slope=coefficients[0]
    intercept=coefficients[1]
    regression=slope*time_history+intercept

    # Figure with certain observables, some of them unused.
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), dpi=300, sharex=True)

    axs[0].plot(time_history, r_mean_history, 'r-', linewidth=2, label=r"NQS $\langle r \rangle$")
    axs[0].plot(times_te, radius_te, 'k--', alpha=0.7, label="Numerical")
    axs[0].set_title(rf"Time evolution with $\gamma = {config['gamma_boost']}$")
    axs[0].set_ylabel(r"$\langle r \rangle$ (fm)")
    axs[0].legend(loc="best")
    axs[0].grid(alpha=0.3)
    # axs[0].set_ylim(1.75, 1.85)

    axs[1].plot(time_history, core_prob_history, 'b-', linewidth=2, label=r"NQS Core ($r < 1$ fm)")
    axs[1].plot(times_te, core_prob_te, 'k--', alpha=0.7, label="Numerical")
    axs[1].set_ylabel("Probability")
    axs[1].legend(loc="best")
    axs[1].grid(alpha=0.3)
    # axs[1].set_ylim(0.76, 0.79)

    axs[2].plot(time_history, overlap_history, 'g-', linewidth=2, label=r"$|\langle \Psi_{NQS}(0) | \Psi_{NQS}(t) \rangle|^2$")
    axs[2].plot(times_te, survival_prob_te, 'k--', alpha=0.7, label="Numerical")
    axs[2].set_ylabel("Overlap")
    axs[2].legend(loc="best")
    axs[2].grid(alpha=0.3)
    # axs[2].set_ylim(0.9999, 1.00001)

    axs[3].plot(time_history, fidelity_history, 'm-', linewidth=2, label=r"Grid Integration $\mathcal{F}$")
    axs[3].plot(time_history, fidelity_mc_history, 'c--', linewidth=1.5, label=r"Walker Integration $\mathcal{F}$")
    axs[3].fill_between(time_history, fidelity_mc_history-fidelity_mc_err_np, fidelity_mc_history+fidelity_mc_err_np, color='cyan', alpha=0.3)
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Fidelity")
    axs[3].legend(loc="best")
    axs[3].grid(alpha=0.3)
    # axs[3].set_ylim(0.998, 1)
    
    plt.tight_layout()

    plt.savefig(os.path.join(run_folder, f"9Observables_evolution.png"), dpi=300)
    plt.close()

    # <r> evolution.
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(time_history, r_mean_history, 'r-', linewidth=2, label=r"NQS $\langle r \rangle$ (Mesh Integral)")
    plt.plot(time_history, r_mean_walkers_history, 'g-', linewidth=2, label=r"NQS $\langle r \rangle$ (Walkers)")
    plt.fill_between(time_history, r_mean_walkers_history-r_mean_walkers_err_history, r_mean_walkers_history+r_mean_walkers_err_history, color='green', alpha=0.3)
    plt.plot(times_te, radius_te, 'k--', alpha=0.7, label="Numerical")
    plt.title(rf"Time evolution with $\gamma = {config['gamma_boost']}$")
    plt.ylabel(r"$\langle r \rangle$ (fm)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.ylim(0.51, 0.55)
    
    plt.savefig(os.path.join(run_folder, f"10Radius_evolution.png"), dpi=300)

    plt.close()

    # Fidelity evolution.
    plt.figure(figsize=(4.6, 3.4), dpi=300)

    # Walker fidelity.
    plt.plot(
        time_history, 
        fidelity_mc_history, 
        color='teal', 
        linewidth=1.5, 
        label=r'Fidelity $\mathcal{F}$'
    )
    
    # Statistical error.
    plt.fill_between(
        time_history, 
        fidelity_mc_history - fidelity_mc_err_np, 
        fidelity_mc_history + fidelity_mc_err_np, 
        color='teal', 
        alpha=0.3
    )

    # Grid integration fidelity.
    plt.plot(
        time_history, 
        fidelity_history, 
        color='red', 
        linewidth=1.0, 
        label=r'Fidelity $\mathcal{F}$ (Grid)'
    )

    plt.xlim(0, time_history[-1])
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\mathcal{F}$')

    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    
    plt.savefig(os.path.join(run_folder, "11_fidelity.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(run_folder, "fidelity.pdf"), bbox_inches='tight')
    plt.close()

    # Energy evolution.
    plt.figure(figsize=(4.6, 3.4), dpi=300)

    std_E_analytical = E_err.item() 

    # <E>.
    plt.plot(
        time_history, 
        energy_history,
        color='purple',
        label=r'$\langle E \rangle$',
        linewidth=1.5
    )

    # Statistical error.
    plt.fill_between(
        time_history,
        energy_history - energy_err_np,
        energy_history + energy_err_np,
        color='purple',
        alpha=0.3,
        label='Statistical error'
    )

    # Linear regression.
    plt.plot(
        time_history,
        regression,
        'r--',
        linewidth=0.8,
        alpha=0.7,
        label=f'Linear fit, slope = {slope:.2e}'
    )

    plt.xlim(0, time_history[-1])
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\langle E \rangle$ (MeV)')
    
    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    
    plt.savefig(os.path.join(run_folder, "12_energy.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(run_folder, "energy.pdf"), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
