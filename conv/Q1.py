import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

# --- Dados do material (curva B-H) ---
H = np.array([0, 68, 135, 203, 271, 338, 406, 474, 542, 609, 1100, 1500, 2500, 4000, 5000, 9000, 12000, 20000, 25000])
B = np.array([0, 0.733, 1.205, 1.424, 1.517, 1.560, 1.588, 1.617, 1.631, 1.646, 1.689, 1.703, 1.724, 1.731, 1.738, 1.761, 1.770, 1.800, 1.816])
B_space = np.linspace(0, B[-1], 100)

# Spline para H(B)
HxB_real = CubicSpline(B, H, bc_type='natural')
HxB_lin = CubicSpline(B[:2], H[:2], bc_type='natural')  # Linearização inicial

# --- Parâmetros físicos ---
N = 1000
g = 2e-3
A = (4e-2)**2
l = 70e-2
u0 = 4e-7 * np.pi
dmax = 20e-3
Npts = 100
x_vals = np.linspace(0, dmax, Npts)

# --- Função principal ---
def calc_forca(I: float):
    lambda_vals = N * A * B_space

    # Matrizes para diferentes modelos de corrente
    i_real, i_lin, i_ideal = [], [], []

    for x in x_vals:
        denom = ((g + x) / u0) * B_space
        i_real.append(((l * HxB_real(B_space)) + denom) / N)
        i_lin.append(((l * HxB_lin(B_space)) + denom) / N)
        i_ideal.append(denom / N)

    i_real = np.array(i_real)
    i_lin = np.array(i_lin)
    i_ideal = np.array(i_ideal)

    # Geração das splines λ(i)
    F_real = [CubicSpline(i_real[i], lambda_vals, bc_type='natural') for i in range(Npts)]
    F_lin  = [CubicSpline(i_lin[i], lambda_vals, bc_type='natural') for i in range(Npts)]
    F_ideal= [CubicSpline(i_ideal[i], lambda_vals, bc_type='natural') for i in range(Npts)]

    # Coenergia e força por modelo
    wc_real, wc_lin, wc_ideal = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)

    for i in range(Npts):
        wc_real[i], _ = quad(F_real[i], 0, I)
        wc_lin[i], _  = quad(F_lin[i], 0, I)
        wc_ideal[i], _= quad(F_ideal[i], 0, I)

    # Derivada da coenergia -> força
    F_r = CubicSpline(x_vals, wc_real).derivative()(x_vals)
    F_l = CubicSpline(x_vals, wc_lin).derivative()(x_vals)
    F_i = CubicSpline(x_vals, wc_ideal).derivative()(x_vals)

    # --- Plot elegante ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals*1000, F_r, label='Real', color='#1f77b4', lw=2)
    plt.plot(x_vals*1000, F_l, label='Linear', linestyle='--', color='#ff7f0e', lw=2)
    plt.plot(x_vals*1000, F_i, label='Ideal', linestyle=':', color='#d62728', lw=2)
    plt.title(f'Força vs Posição para I = {I} A', fontsize=14)
    plt.xlabel('Posição do êmbolo (mm)', fontsize=12)
    plt.ylabel('Força (N)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Execução ---
for corrente in [1, 5, 15]:
    calc_forca(corrente)

# Cálculo da corrente máxima permitida (com B máx e x=5mm)
H_max = HxB_real(B[-1])
I_max = ((l * H_max) + ((g + 5e-3) / u0) * B[-1]) / N
print(f"Corrente máxima estimada: {I_max:.2f} A")
