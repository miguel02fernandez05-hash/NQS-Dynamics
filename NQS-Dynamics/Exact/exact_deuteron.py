# Exact numerical Deuteron

import numpy as np
from scipy.sparse import diags
# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch

hbar_c = 197.32698  # MeV·fm.
m_p = 938.272088    # Proton mass, MeV/c^2.
m_n = 939.565420   # Neutron mass, MeV/c^2.
mass_factor=1
m_p_eff=mass_factor*m_p
m_n_eff=mass_factor*m_n

# Reduced mass.
mu = (m_p_eff*m_n_eff)/(m_p_eff+m_n_eff)
kin_prefactor = (hbar_c**2)/(2*mu)  # hbar·c/(2·mu)   MeV*fm^2.

# LO pionless EFT consants.
C_10 = -142  # MeV
R_0 = 1  # fm.
gamma = 0.00075
dt = 0.01
time_steps = 40000 
save_interval = 120

r_max = 100   # Max r (in fm). A large enough value of this is needed to obtain good results.
N = 1250000 
dr = r_max / N
r = np.linspace(dr, r_max, N)

# Build Hamiltonian matrix.
diag = np.ones(N)*(2.0/dr**2)
off_diag = np.ones(N-1)*(-1/dr**2)

# Kinetic energy matrix.
# [-1, 0, -1] is the "offset".
T = kin_prefactor*diags([off_diag, diag, off_diag], [-1, 0, 1])

# Potential energy matrix.
V = C_10*np.exp(-(r/R_0)**2)

# Kinetic (2/dr^2) + Potential V(r).
main_diag = kin_prefactor*(2/dr**2)*np.ones(N)+V

# Kinetic (-1/dr^2).
off_diag = kin_prefactor*(-1/dr**2)*np.ones(N-1)

# Ground state (index 0).
eigenvalues, eigenvectors = eigh_tridiagonal(
    main_diag,
    off_diag,
    select='i',
    select_range=(0, 0))

E_exact = eigenvalues[0]
u_exact = eigenvectors[:, 0]

# Positive solution by checking the value at the middle of the grid.
if u_exact[N//2] < 0:
    u_exact = -u_exact

# u(r)=r·psi(r).
psi_exact_1D = u_exact/r

# 3D normalization.
norm_factor = np.sqrt(np.sum((psi_exact_1D**2)*4*np.pi*(r**2)*dr))
psi_exact = psi_exact_1D/norm_factor

print(f"Exact Binding Energy: {E_exact:.4f} MeV")

# <r>.
# r·|psi|^2·4·pi*r^2·dr
r_expected_exact = np.sum(r*(psi_exact**2)*4*np.pi*(r**2)*dr)

print(f"Exact <r>: {r_expected_exact:.4f} fm")

# \sqrt{<r^2>}.
r2_expected_exact = np.sqrt(np.sum((r**2)*(psi_exact**2)*4*np.pi*(r**2)*dr))/2
print(f"Exact sqrt(<r^2>): {r2_expected_exact:.4f} fm")

filename = f"exact_deuteron_target_C_10_142_Mass_Factor_{mass_factor}.npz"
np.savez(filename,
         r=r,
         psi=psi_exact,
         E=E_exact,
         C_10=C_10,
         R_0=R_0,
         r_expected_exact=r_expected_exact,
         r2_expected_exact=r2_expected_exact)


# Since when doing integrals we sum over r including 4·pi·r, normalizing u(r) is the same as normalizing psi(r).
norm_u = np.sqrt(np.sum(np.abs(u_exact)**2*dr))
u_normalized = (u_exact/norm_u).astype(complex)
u_initial = np.copy(u_normalized)
u_boosted = u_normalized*np.exp(1j*gamma*r**2)
u_initial_ak = np.copy(u_boosted)  # After kick.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# By forcing u(-r)=-u(r), we make sure that u(r=0)=0 (since u(r)=r*psi(r)).
# -r[::-1] goes from the last value of r to the first, with a minus sign. Builds [-R_max,...,R_max]
r_mirror = np.concatenate([-r[::-1], r])
# Same but with the wavefunction.
u_mirror = np.concatenate([-u_boosted[::-1], u_boosted])

V_mirror = np.concatenate([V[::-1], V])

N_mirror = len(r_mirror)
L_total = 2*r_max
dk = 2*np.pi/L_total
# Grid from -N_mirror/2 to +N_mirror/2.
k_linear = (np.arange(N_mirror)-N_mirror/2)*dk
K_op_mirror = kin_prefactor*k_linear**2

# Time evolution operators.
U_V_m = np.exp(-1j*V_mirror*(dt/2)/hbar_c)
U_K_m = np.exp(-1j*K_op_mirror*dt/hbar_c)

zi = complex(0, 1)
alt_sign_fwd = np.exp(-zi*np.pi*np.arange(N_mirror))
alt_sign_inv = np.exp(+zi*np.pi*np.arange(N_mirror))


u_mirror_t = torch.tensor(u_mirror, dtype=torch.complex128, device=device)
U_V_m_t = torch.tensor(U_V_m, dtype=torch.complex128, device=device)
U_K_m_t = torch.tensor(U_K_m, dtype=torch.complex128, device=device)

alt_sign_fwd_t = torch.tensor(alt_sign_fwd, dtype=torch.complex128, device=device)
alt_sign_inv_t = torch.tensor(alt_sign_inv, dtype=torch.complex128, device=device)


r_t = torch.tensor(r, dtype=torch.float64, device=device)
u_initial_ak_t = torch.tensor(u_initial_ak, dtype=torch.complex128, device=device)
core_mask = r < 1
core_mask_t = torch.tensor(core_mask, dtype=torch.bool, device=device)

num_saves = time_steps // save_interval + 1
times_t = torch.zeros(num_saves, dtype=torch.float64, device=device)
# r2_history_t = torch.zeros(num_saves, dtype=torch.float64, device=device)
r_history_t = torch.zeros(num_saves, dtype=torch.float64, device=device)
core_prob_history_t = torch.zeros(num_saves, dtype=torch.float64, device=device)
survival_history_t = torch.zeros(num_saves, dtype=torch.float64, device=device)
u_history_t = torch.zeros((num_saves, N), dtype=torch.complex128, device=device)

save_idx = 0

# Observables at t=0.
u_current_t = u_mirror_t[N:]
prob_density_t = torch.abs(u_current_t)**2

r_exp_t = torch.sum(r_t * prob_density_t * dr)
core_prob_t = torch.sum(prob_density_t[core_mask_t] * dr)

overlap_t = torch.sum(torch.conj(u_initial_ak_t) * u_current_t * dr)
survival_prob_t = torch.abs(overlap_t)**2

times_t[save_idx] = 0.0
r_history_t[save_idx] = r_exp_t
core_prob_history_t[save_idx] = core_prob_t
survival_history_t[save_idx] = survival_prob_t
u_history_t[save_idx] = u_current_t

save_idx += 1

for step in tqdm(range(1,time_steps+1)):
    # Half step in position space.
    u_mirror_t = U_V_m_t * u_mirror_t

    # Fourier transform on GPU.
    u_k_t = alt_sign_fwd_t * torch.fft.fft(alt_sign_fwd_t * u_mirror_t)

    # Step in momentum space.
    u_k_t = U_K_m_t * u_k_t

    # Inverse fourier transform on GPU.
    u_mirror_t = alt_sign_inv_t * torch.fft.ifft(alt_sign_inv_t * u_k_t)

    # Half step in position space.
    u_mirror_t = U_V_m_t * u_mirror_t

    if step % save_interval == 0:
        u_current_t = u_mirror_t[N:]
        prob_density_t = torch.abs(u_current_t)**2

        # r2_exp_t = torch.sum((r_t**2) * prob_density_t * dr)
        # rms_radius_t = torch.sqrt(r2_exp_t) / 2

        r_exp_t = torch.sum(r_t * prob_density_t * dr)

        core_prob_t = torch.sum(prob_density_t[core_mask_t] * dr)

        overlap_t = torch.sum(torch.conj(u_initial_ak_t) * u_current_t * dr)
        survival_prob_t = torch.abs(overlap_t)**2

        times_t[save_idx] = step * dt
        # r2_history_t[save_idx] = rms_radius_t
        r_history_t[save_idx] = r_exp_t
        core_prob_history_t[save_idx] = core_prob_t
        survival_history_t[save_idx] = survival_prob_t
        u_history_t[save_idx] = u_current_t
        
        save_idx += 1


times_np = times_t.cpu().numpy()
# r2_history_np = r2_history_t.cpu().numpy()
r_history_np = r_history_t.cpu().numpy()
core_prob_history_np = core_prob_history_t.cpu().numpy()
survival_history_np = survival_history_t.cpu().numpy()
u_history_np = u_history_t.cpu().numpy()

np.savez(f"exact_time_evolution_{gamma}_Mass_Factor_{mass_factor}.npz",
         times=times_np,
         r=r,
        #  rms_radius=r2_history_np,
         r_expected=r_history_np,
         core_prob=core_prob_history_np,
         survival_prob=survival_history_np,
         u_history=u_history_np)


fig, axs = plt.subplots(3, 1, figsize=(8, 10), dpi=300, sharex=True)

axs[0].plot(times_np, r_history_np, 'r-', linewidth=2, label=r"$\langle r \rangle$")
axs[0].axhline(y=r_expected_exact, color='k', linestyle='--',
               alpha=0.5, label=r"Initial GS $\langle r \rangle$")
axs[0].set_title(rf"Deuteron kicked ($\gamma={gamma}$)")
axs[0].set_ylabel(r"$\langle r \rangle$ (fm)")
axs[0].legend(loc="upper left")
axs[0].grid(alpha=0.3)

axs[1].plot(times_np, core_prob_history_np, 'b-', linewidth=2,
            label=r"Probability in Core ($r < 1$ fm)")
axs[1].set_ylabel("Probability")
axs[1].legend(loc="upper right")
axs[1].grid(alpha=0.3)

axs[2].plot(times_np, survival_history_np, 'g-', linewidth=2,
            label=r"$|\langle \psi(0) | \psi(t) \rangle|^2$")
axs[2].set_xlabel("Time (fm/c)")
axs[2].set_ylabel("Survival Prob.")
axs[2].legend(loc="upper right")
axs[2].grid(alpha=0.3)

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, f"Exact_observables_evolution_gamma_{gamma}_Mass_Factor_{mass_factor}.png"), dpi=300)