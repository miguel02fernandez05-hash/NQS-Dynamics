# main_QHO.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from ground_state_QHO import ComplexHarmonicNQS, train_ground_state
from time_evolution_QHO import run_time_evolution

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

def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    config = {
        'device': device,
        'Nin': 3, 'Nout': 1, 'Nhid': 20,
        'seed': 1,
        'pretraining_epochs': 0,
        'epochs': 400,
        'lr': 0.01,
        'epsilonSR': 3e-5,
        'step_size_gs': 0.7,
        'n_walkers': 50000,
        'Nx': 200,
        'train_a': -10, 'train_b': 10,
        'wx': 1, 'wy': 1, 'wz': 1,
        'x0': 0, 'y0': 0, 'z0': 0,
        'x0p': 1, 'y0p': 1, 'z0p': 1,
        'kick_x': 1, 'kick_y': 1, 'kick_z': 1,
        'dt': 0.01,
        'total_time': 10,
        'step_sizeRK':0.7,
        'epsilonRK':7e-6    # EpsilonTDVP.
    }
    
    config['h'] = (config['train_b']-config['train_a'])/(config['Nx']-1)
    config['dV'] = config['h']**3

    # Mesh for grid calculations.
    coords = torch.linspace(config['train_a'], config['train_b'], config['Nx'])
    x_mesh, y_mesh, z_mesh = torch.meshgrid(coords, coords, coords, indexing='ij')
    inputs_flat = torch.stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()], dim=1)
    X_train = inputs_flat.to(config['device']).requires_grad_(True)
    coords_plot = coords.numpy()

    base_filename = (f"Nhid_{config['Nhid']}_"
                     f"Walkers_{config['n_walkers']}_"
                     f"dt_{config['dt']}_"
                     f"epsilonSR_{config['epsilonSR']}_"
                     f"step_sizeRK_{config['step_sizeRK']}_"
                     f"epsilonRK_{config['epsilonRK']}_"
                     f"wx_{config['wx']}_wy_{config['wy']}_wz_{config['wz']}_"
                     f"x0_{config['x0p']}_y0_{config['y0p']}_z0_{config['z0p']}_"
                     f"kickx_{config['kick_x']}_kicky_{config['kick_y']}_kickz_{config['kick_z']}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Folder where the plots are saved.
    run_folder = os.path.join(script_dir, "Plots_QHO", base_filename)
    os.makedirs(run_folder, exist_ok=True)

    # Initial value of the weight matrices and bias vectors.
    torch.manual_seed(config['seed'])

    W1_amp = torch.rand(config['Nhid'], config['Nin'], requires_grad=True) * (-1.)
    B1_amp = torch.rand(config['Nhid'], requires_grad=True) * 2. - 1.
    W2_amp = torch.rand(config['Nhid'], config['Nhid'], requires_grad=True) * (-1.) 
    B2_amp = torch.rand(config['Nhid'], requires_grad=True) * 2. - 1.
    W3_amp = torch.rand(config['Nout'], config['Nhid'], requires_grad=True)

    W1_phase = torch.rand(config['Nhid'], config['Nin'], requires_grad=True) * (-1.)
    B1_phase = torch.rand(config['Nhid'], requires_grad=True) * 2. - 1.
    W2_phase = torch.rand(config['Nhid'], config['Nhid'], requires_grad=True) * (-1.)
    B2_phase = torch.rand(config['Nhid'], requires_grad=True) * 2. - 1.
    W3_phase = torch.rand(config['Nout'], config['Nhid'], requires_grad=True)

    freqs_sq = torch.tensor([config['wx']**2, config['wy']**2, config['wz']**2], device=config['device'])
    shifts_GS = torch.tensor([config['x0'], config['y0'], config['z0']], device=config['device'])
    shifts_TE = torch.tensor([config['x0p'], config['y0p'], config['z0p']], device=config['device'])

    # Target wavefunction.
    E_true = 0.5 * (config['wx'] + config['wy'] + config['wz'])
    exponent = -0.5 * (config['wx'] * (X_train[:, 0] - config['x0'])**2 + 
                       config['wy'] * (X_train[:, 1] - config['y0'])**2 + 
                       config['wz'] * (X_train[:, 2] - config['z0'])**2)
    normalization = (config['wx'] * config['wy'] * config['wz'] / np.pi**3)**(1/4)
    target = normalization * torch.exp(exponent)

    net = ComplexHarmonicNQS(
        config['Nin'], config['Nout'], config['Nhid'], W1_amp, B1_amp, W2_amp, B2_amp, W3_amp, 
        W1_phase, B1_phase, W2_phase, B2_phase, W3_phase, config['x0'], config['y0'], config['z0']
    ).to(config['device'])

    # This would be used for pretraining on a grid, but pretraining_epochs is set to 0 because it's not needed.
    optimizer = torch.optim.Adam(params=net.parameters(), lr=config['lr'])
    


    # This callback was originally used for local interactive runs, where the
    # training plot could be refreshed while the ground-state optimization was
    # still running. On Artemisa, there is no interactive display, so the live
    # refresh is disabled by keeping plt.pause(0.01) commented out inside the
    # callback. In order to bring back the interactive feature that updates the
    # figure while the code is running, uncomment the line plt.pause(0.01).


    plt.ion()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5.8, 5.2))
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Function to update figure while code is still running.
    def pic_callback(epoch, psi_mesh, psi_norm_mesh, loss_accum, U_accum, K_accum):
        psi_norm_reshaped = np.abs(psi_norm_mesh.detach().cpu().numpy()).reshape(
            config['Nx'], config['Nx'], config['Nx'])
        
        target_reshaped = target.detach().cpu().numpy().reshape(config['Nx'], config['Nx'], config['Nx'])

        idx_x0 = (np.abs(coords_plot-config['x0'])).argmin()
        idx_y0 = (np.abs(coords_plot-config['y0'])).argmin()
        idx_z0 = (np.abs(coords_plot-config['z0'])).argmin()

        slice_x_nqs = psi_norm_reshaped[:, idx_y0, idx_z0]
        slice_y_nqs = psi_norm_reshaped[idx_x0, :, idx_z0]
        slice_z_nqs = psi_norm_reshaped[idx_x0, idx_y0, :]
        
        slice_x_tar = target_reshaped[:, idx_y0, idx_z0]
        slice_y_tar = target_reshaped[idx_x0, :, idx_z0]
        slice_z_tar = target_reshaped[idx_x0, idx_y0, :]

        def update_plot(ax, x_data, y_nqs, y_tar, title):
            max_val = max(np.max(y_nqs), np.max(y_tar))
            ax.set_ylim(-0.05, max_val*1.2 if max_val > 0 else 1)
            ax.set_title(title)
            if ax.lines:
                ax.lines[0].set_data(x_data, y_nqs)
                ax.lines[1].set_data(x_data, y_tar)
            else:
                ax.plot(x_data, y_nqs, label=r'$\Psi_{NQS}$', color='b')
                ax.plot(x_data, y_tar, '--', label=r'$\Psi_{Target}$', color='r')
                ax.legend(loc='upper right', fontsize='small')

            
        def fmt_coord(value):
            return f"{float(value):g}"

        x0_label = fmt_coord(config["x0"])
        y0_label = fmt_coord(config["y0"])
        z0_label = fmt_coord(config["z0"])

        update_plot(ax1, coords_plot, slice_x_nqs, slice_x_tar, r"Wavefunction on X-axis")
        update_plot(ax2, coords_plot, slice_y_nqs, slice_y_tar, r"Wavefunction on Y-axis")
        update_plot(ax3, coords_plot, slice_z_nqs, slice_z_tar, r"Wavefunction on Z-axis")

        ax1.set_xlabel("x")
        ax2.set_xlabel("y")
        ax3.set_xlabel("z")

        ax1.set_ylabel(rf"$\Psi(x, {y0_label}, {z0_label})$")
        ax2.set_ylabel(rf"$\Psi({x0_label}, y, {z0_label})$")
        ax3.set_ylabel(rf"$\Psi({x0_label}, {y0_label}, z)$")

        x_epochs = np.arange(1, epoch+2)
        y_loss = loss_accum[:epoch+1].detach().cpu().numpy()
        y_U = U_accum[:epoch+1].detach().cpu().numpy()
        y_K = K_accum[:epoch+1].detach().cpu().numpy()

        E_virial=E_true/2

        ax4.cla()
        ax4.set_xlim(0, epoch+1)
        ax4.set_ylim(0, E_true * 1.5) 
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Energy")
        ax4.set_title("Energy")
        ax4.plot(x_epochs, y_loss, label='$E$', color='b', linewidth=1.2)
        ax4.plot(x_epochs, y_U, label='$U$', color='orange', linewidth=0.8, alpha=0.8)
        ax4.plot(x_epochs, y_K, label='$K$', color='green', linewidth=0.8, alpha=0.8)
        
        ax4.plot(x_epochs, np.ones_like(x_epochs)*E_true, '--', label=rf'$E_0 = {E_true:.2f}$', color='k', alpha=0.6)
        ax4.plot(x_epochs, np.ones_like(x_epochs)*E_virial, ':', label='$U_0, K_0$', color='k', alpha=0.6)
        ax4.legend(loc='lower right', fontsize='small')

        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
        ax3.grid(alpha=0.3)
        ax4.grid(alpha=0.3)
        
        # plt.pause(0.01)

    print("NN architecture:\n", net)    

    # Ground state optimization.

    net, walkers, loss_accum, walkers_init, E_var, E_err = train_ground_state(
        net=net, optimizer=optimizer, X_train=X_train, coords_plot=coords_plot, 
        freqs_sq=freqs_sq, shifts=shifts_GS, pretraining_epochs=config['pretraining_epochs'], 
        epochs=config['epochs'], n_walkers=config['n_walkers'], dV=config['dV'], lr=config['lr'], 
        target=target, wx=config['wx'], wy=config['wy'], wz=config['wz'], 
        x0=config['x0'], y0=config['y0'], z0=config['z0'], 
        device=config['device'], Nx=config['Nx'], epsilon=config['epsilonSR'], plot_callback=pic_callback)

    training_plot_path = os.path.join(run_folder, "0_training.png")
    training_plot_path_pdf = os.path.join(run_folder, "0_training.pdf")

    fig.savefig(training_plot_path, bbox_inches='tight', dpi=300)
    fig.savefig(training_plot_path_pdf, bbox_inches='tight')

    plt.ioff()
    plt.close(fig)

    E0 = loss_accum[-1]
    print(f"\nFinal Ground State Energy: {E0:.4f} ± {E_err:.4f}")
    print(f"Energy Variance: {E_var:.4f}")

    # Initial and final distribution of the walkers in the ground state optimization.

    walkers_init_np = walkers_init.cpu().numpy()
    walkers_final_np = walkers.detach().cpu().numpy()

    # Ground state optimization figure.

    fig_init, ax_init = plt.subplots(figsize=(5.8, 3.5), dpi=300)
    ax_init.hist(walkers_init_np[:, 0], bins=80, density=True, alpha=0.6, label='X init')
    ax_init.hist(walkers_init_np[:, 1], bins=80, density=True, alpha=0.6, label='Y init')
    ax_init.hist(walkers_init_np[:, 2], bins=80, density=True, alpha=0.6, label='Z init')
    ax_init.set_title("Initial Walkers Distribution (After Thermalization)")
    ax_init.set_xlabel("Position")
    ax_init.set_ylabel("Density")
    ax_init.grid(alpha=0.3)
    ax_init.legend()
    fig_init.savefig(os.path.join(run_folder, "0a_initial_walkers.png"), bbox_inches='tight', dpi=300)
    fig_init.savefig(os.path.join(run_folder, "0a_initial_walkers.pdf"), bbox_inches='tight')
    plt.close(fig_init)

    fig_final, ax_final = plt.subplots(figsize=(5, 3.5), dpi=300)
    ax_final.hist(walkers_final_np[:, 0], bins=80, density=True, alpha=0.6, color='C0', label='Walkers X coord')
    ax_final.hist(walkers_final_np[:, 1], bins=80, density=True, alpha=0.6, color='C1', label='Walkers Y coord')
    ax_final.hist(walkers_final_np[:, 2], bins=80, density=True, alpha=0.6, color='C2', label='Walkers Z coord')
    ax_final.grid(alpha=0.3)

    grid = np.linspace(config['train_a'], config['train_b'], 1000)
    exact_x = np.sqrt(config['wx'] / np.pi) * np.exp(-config['wx'] * (grid - config['x0'])**2)
    exact_y = np.sqrt(config['wy'] / np.pi) * np.exp(-config['wy'] * (grid - config['y0'])**2)
    exact_z = np.sqrt(config['wz'] / np.pi) * np.exp(-config['wz'] * (grid - config['z0'])**2)
    
    def fmt_coord(value):
        return f"{float(value):g}"

    x0_label = fmt_coord(config["x0"])
    y0_label = fmt_coord(config["y0"])
    z0_label = fmt_coord(config["z0"])

    ax_final.plot(grid, exact_x, color='C0', label=rf'$|\Psi(x, {y0_label}, {z0_label})|^2$')
    ax_final.plot(grid, exact_y, color='C1', label=rf'$|\Psi({x0_label}, y, {z0_label})|^2$')
    ax_final.plot(grid, exact_z, color='C2', label=rf'$|\Psi({x0_label}, {y0_label}, z)|^2$')

    ax_final.set_title("Target vs Sampled Densities")
    ax_final.set_xlabel("Position")
    ax_final.set_ylabel("Probability Density")
    ax_final.legend(fontsize='small', loc='upper right')
    ax_final.set_xlim(-6,6)
    fig_final.savefig(os.path.join(run_folder, "0b_final_walkers.png"), bbox_inches='tight', dpi=300)
    fig_final.savefig(os.path.join(run_folder, "0b_final_walkers.pdf"), bbox_inches='tight')
    
    plt.close(fig_final)

    # Time evolution.

    # Introduce the kick.
    kick_tensor = torch.tensor([config['kick_x'], config['kick_y'], config['kick_z']], device=config['device'])

    print("\nStarting Time Evolution...")
    histories = run_time_evolution(
        net=net, walkers=walkers, dt=config['dt'], total_time=config['total_time'], 
        n_walkers=config['n_walkers'], freqs_sq=freqs_sq, shifts_TE=shifts_TE, shifts_GS=shifts_GS, 
        device=config['device'], X_train=X_train, kick=kick_tensor, step_sizeRK=config['step_sizeRK'], epsilon=config['epsilonRK']
    )

    t_hist = histories['t_hist']
    e_hist = histories['e_hist']
    p_hist = histories['p_hist']
    s_hist = histories['s_hist']
    f_hist = histories['f_hist']
    err_f_hist = histories['err_f_hist']
    f_grid_hist = histories['f_grid_hist']
    std_e_hist = histories['std_e_hist']

    # Linear regression in energy figure.

    coefficients = np.polyfit(t_hist, e_hist, 1)
    slope, intercept = coefficients[0], coefficients[1]
    regression = slope * t_hist + intercept

    d_x = (config['x0'] - config['x0p']) * np.cos(config['wx'] * t_hist)+(config['kick_x'] / config['wx']) * np.sin(config['wx'] * t_hist)
    d_y = (config['y0'] - config['y0p']) * np.cos(config['wy'] * t_hist)+(config['kick_y'] / config['wy']) * np.sin(config['wy'] * t_hist)
    d_z = (config['z0'] - config['z0p']) * np.cos(config['wz'] * t_hist)+(config['kick_z'] / config['wz']) * np.sin(config['wz'] * t_hist)

    uncertainty_E_analytical = np.sqrt(0.5*(config['wx']**3*d_x**2+config['wy']**3*d_y**2+config['wz']**3*d_z**2))/np.sqrt(config['n_walkers'])

    exact_evo_x = config['x0p'] + (config['x0'] - config['x0p']) * np.cos(config['wx'] * t_hist) + (config['kick_x'] / config['wx']) * np.sin(config['wx'] * t_hist)
    exact_evo_y = config['y0p'] + (config['y0'] - config['y0p']) * np.cos(config['wy'] * t_hist) + (config['kick_y'] / config['wy']) * np.sin(config['wy'] * t_hist)
    exact_evo_z = config['z0p'] + (config['z0'] - config['z0p']) * np.cos(config['wz'] * t_hist) + (config['kick_z'] / config['wz']) * np.sin(config['wz'] * t_hist)
    
    # Format for the figures.
    single_figsize = (4.6, 3.4)
    single_dpi = 300

    # <r>.

    fig_pos, ax_pos = plt.subplots(figsize=single_figsize, dpi=single_dpi)

    ax_pos.plot(t_hist, p_hist[:, 0], label=r'$\langle x \rangle$')
    ax_pos.plot(t_hist, p_hist[:, 1], label=r'$\langle y \rangle$')
    ax_pos.plot(t_hist, p_hist[:, 2], label=r'$\langle z \rangle$')

    ax_pos.plot(
        t_hist, exact_evo_x, '--',
        color='black', alpha=0.7, linewidth=1,
        label=r'Exact'
    )
    ax_pos.plot(t_hist, exact_evo_y, '--', color='black', alpha=0.7, linewidth=1)
    ax_pos.plot(t_hist, exact_evo_z, '--', color='black', alpha=0.7, linewidth=1)

    ax_pos.set_xlabel(r'Time $t$')
    ax_pos.set_ylabel(r'$\langle \mathbf{r} \rangle$')
    ax_pos.legend(loc='best', fontsize=8)
    ax_pos.grid(alpha=0.3)

    fig_pos.tight_layout()

    evo_plot_path = os.path.join(run_folder, "position.png")
    evo_plot_path_pdf = os.path.join(run_folder, "position.pdf")
    fig_pos.savefig(evo_plot_path, bbox_inches='tight', dpi=300)
    fig_pos.savefig(evo_plot_path_pdf, bbox_inches='tight')

    plt.close(fig_pos)

    # <E>.

    fig_energy, ax_energy = plt.subplots(figsize=single_figsize, dpi=single_dpi)

    ax_energy.plot(
        t_hist, e_hist,
        color='purple',
        label=r'$\langle E \rangle$'
    )

    # Statistical error.
    ax_energy.fill_between(
        t_hist,
        e_hist - std_e_hist,
        e_hist + std_e_hist,
        color='purple',
        alpha=0.3,
        label='Statistical error'
    )

    # Analytical uncertainty in energy.
    ax_energy.plot(
        t_hist,
        regression + uncertainty_E_analytical,
        'k--',
        linewidth=1,
        label='Analytical statistical error'
    )

    ax_energy.plot(
        t_hist,
        regression - uncertainty_E_analytical,
        'k--',
        linewidth=1
    )

    # Linear regression.
    ax_energy.plot(
        t_hist,
        regression,
        'r--',
        linewidth=0.8,
        alpha=0.7,
        label=f'Linear fit, slope = {slope:.2e}'
    )

    ax_energy.set_xlabel(r'Time $t$')
    ax_energy.set_ylabel(r'$\langle E \rangle$')
    ax_energy.legend(loc='best', fontsize=8)
    ax_energy.grid(alpha=0.3)

    fig_energy.tight_layout()

    evo_plot_path = os.path.join(run_folder, "energy.png")
    evo_plot_path_pdf = os.path.join(run_folder, "energy.pdf")
    fig_energy.savefig(evo_plot_path, bbox_inches='tight', dpi=300)
    fig_energy.savefig(evo_plot_path_pdf, bbox_inches='tight')

    plt.close(fig_energy)

    # Fidelity.

    fig_fid, ax_fid = plt.subplots(figsize=single_figsize, dpi=single_dpi)

    ax_fid.plot(
        t_hist,
        f_hist,
        color='teal',
        linewidth=1.5,
        label=r'Fidelity $\mathcal{F}$'
    )

    ax_fid.fill_between(
        t_hist,
        f_hist - err_f_hist,
        f_hist + err_f_hist,
        color='teal',
        alpha=0.3
    )

    ax_fid.plot(
        t_hist,
        f_grid_hist,
        color='red',
        linewidth=1,
        label=r'Fidelity $\mathcal{F}$ (Grid)'
    )

    ax_fid.set_xlabel(r'Time $t$')
    ax_fid.set_ylabel(r'$\mathcal{F}$')

    f_min = min(np.min(f_hist - err_f_hist), np.min(f_grid_hist))
    f_max = max(np.max(f_hist + err_f_hist), np.max(f_grid_hist))
    margin = 0.05 * (f_max - f_min) if f_max > f_min else 5e-4
    ax_fid.set_ylim(f_min - margin, f_max + margin)

    ax_fid.legend(loc='best', fontsize=8)
    ax_fid.grid(alpha=0.3)

    fig_fid.tight_layout()

    evo_plot_path = os.path.join(run_folder, "fidelity.png")
    evo_plot_path_pdf = os.path.join(run_folder, "fidelity.pdf")
    fig_fid.savefig(evo_plot_path, bbox_inches='tight', dpi=300)
    fig_fid.savefig(evo_plot_path_pdf, bbox_inches='tight')

    plt.close(fig_fid)

    # All plots in the same figure.

    fig2, axs2 = plt.subplots(3, 1, figsize=(5.8, 6.8), dpi=300, sharex=True)
    
    axs2[0].plot(t_hist, p_hist[:, 0], label=r'$\langle x \rangle$')
    axs2[0].plot(t_hist, p_hist[:, 1], label=r'$\langle y \rangle$')
    axs2[0].plot(t_hist, p_hist[:, 2], label=r'$\langle z \rangle$')
    axs2[0].plot(t_hist, exact_evo_x, '--', color='black', alpha=0.7, linewidth=1, label=r'$\langle r \rangle_{Exact}$')
    axs2[0].plot(t_hist, exact_evo_y, '--', color='black', alpha=0.7, linewidth=1)
    axs2[0].plot(t_hist, exact_evo_z, '--', color='black', alpha=0.7, linewidth=1)
    axs2[0].set_ylabel(r'$\langle \mathbf{r} \rangle$')
    axs2[0].legend(loc='upper right')
    axs2[0].grid(alpha=0.3)

    axs2[1].plot(t_hist, e_hist, color='purple', label='Energy')
    axs2[1].fill_between(t_hist, e_hist - std_e_hist, e_hist + std_e_hist, color='purple', alpha=0.3, label='Statistical Error')
    axs2[1].plot(t_hist, regression + uncertainty_E_analytical, 'k--', linewidth=1, label='Analytical Statistical Error')
    axs2[1].plot(t_hist, regression - uncertainty_E_analytical, 'k--', linewidth=1)
    axs2[1].plot(t_hist, regression, 'r--', linewidth=0.6, alpha=0.7, label=f'Linear Fit (slope: {slope:.2e})')
    axs2[1].set_ylabel(r'$\langle E \rangle$')
    axs2[1].legend(loc='upper right')
    axs2[1].grid(alpha=0.3)

    axs2[2].plot(t_hist, f_hist, color='teal', linewidth=1.5, label=r'Fidelity $\mathcal{F}$')
    axs2[2].fill_between(t_hist, f_hist - err_f_hist, f_hist + err_f_hist, color='teal', alpha=0.3)
    axs2[2].plot(t_hist, f_grid_hist, color='red', linewidth=1, label=r'Fidelity $\mathcal{F}$ (Grid)')
    axs2[2].set_xlabel('Time $t$')
    axs2[2].set_ylabel(r'$\mathcal{F}$')
    axs2[2].set_ylim(np.min(f_hist)-0.0005, np.max(f_hist)+0.0005)
    axs2[2].legend(loc='upper right')
    axs2[2].grid(alpha=0.3)

    plt.tight_layout()
    evo_plot_path = os.path.join(run_folder, "1_time_evolution.png")
    evo_plot_path_pdf = os.path.join(run_folder, "1_time_evolution.pdf")
    fig2.savefig(evo_plot_path, bbox_inches='tight', dpi=300)
    fig2.savefig(evo_plot_path_pdf, bbox_inches='tight')
    plt.show()
    


if __name__ == "__main__":
    main()