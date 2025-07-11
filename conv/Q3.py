import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import trapezoid

# ===============================
# Estilo gráfico
# ===============================
plt.style.use('seaborn-v0_8-poster')
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "legend.fontsize": 13,
    "figure.figsize": (10, 6),
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.linestyle": ":",
})

# ===============================
# Parâmetros físicos e geométricos
# ===============================
w = h = D = 3e-2
d = 4e-2
g = 0.25e-2
u0 = 4 * np.pi * 1e-7
Hc = -50e3
Br = 1.2
ur = Br / -Hc
Nf_val = 1500
Npts = 100

B_data = np.array([1.20, 1.19, 1.15, 1.13, 1.11, 1.08, 1.05, 1.02,
                   0.96, 0.85, 0.80, 0.75, 0.70, 0.60, 0.10, 0.01])
H_data = np.array([0.00, -5.00, -10.00, -15.00, -20.00, -25.00, -30.00, -35.00,
                   -40.00, -45.00, -46.00, -47.00, -48.00, -49.00, -50.00, -50.75]) * 1e3

interp_H_B_real = interp1d(B_data[::-1], H_data[::-1], kind='cubic', fill_value="extrapolate")
interp_H_B_lin = lambda Bm: (Bm / ur) + Hc

area_ima = w * D
area_ar = lambda x: (h + x) * D

# ===============================
# Funções de visualização (versão bonita)
# ===============================
def plot_H_vs_B():
    B_vals = np.linspace(0, 1.2, 200)
    plt.figure()
    plt.plot(interp_H_B_real(B_vals), B_vals, label="Modelo Real", color="#0077b6")
    plt.plot(interp_H_B_lin(B_vals), B_vals, label="Modelo Linear", linestyle='--', color="#ff6f61")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel("Campo Magnético H (A/m)")
    plt.ylabel("Densidade de Fluxo B (T)")
    plt.title("Curva de Magnetização B-H")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fluxo_vs_corrente(If_r, Bg_r, If_l, Bg_l):
    plt.figure()
    plt.plot(If_r, Bg_r, label="Modelo Real", color="#118ab2", linewidth=1.5, marker='o', markersize=4)
    plt.plot(If_l, Bg_l, label="Modelo Linear", linestyle='--', color="#ef476f", linewidth=1.5, marker='s', markersize=4)
    plt.xlabel("Corrente (A)")
    plt.ylabel("Densidade de Fluxo B_g (T)")
    plt.title("Densidade de Fluxo no Entreferro em x ≈ h/2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forca_vs_posicao(pos_r, F_r, pos_l, F_l):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axs[0].plot(pos_r * 100, np.abs(F_r), color="#073b4c", marker='o', markersize=4,
                linewidth=1.5, label="Real")
    axs[0].set_title("Força Magnética (Modelo Real)")
    axs[0].set_xlabel("Posição (cm)")
    axs[0].set_ylabel("Força (N)")
    axs[0].legend()

    axs[1].plot(pos_l * 100, np.abs(F_l), color="#ef476f", linestyle="--", marker='s',
                markersize=4, linewidth=1.5, label="Linear")
    axs[1].set_title("Força Magnética (Modelo Linear)")
    axs[1].set_xlabel("Posição (cm)")
    axs[1].legend()

    plt.suptitle("Força Magnética em Função da Posição", fontsize=18)
    plt.tight_layout()
    plt.show()

# ===============================
# Cálculo da coenergia e força
# ===============================
def calcular_coenergia(H_interp, Nf):
    posicoes = np.linspace(0, h, Npts)
    Bm_vals = np.linspace(0, 1.2, Npts)
    coenergia = np.zeros(Npts - 1)

    Bg_h2 = None
    If_h2 = None

    for i, x in enumerate(posicoes[:-1]):
        A_ar = area_ar(x)
        If_vals = np.zeros_like(Bm_vals)
        fluxo_vals = np.zeros_like(Bm_vals)

        for j, Bm in enumerate(Bm_vals):
            H = H_interp(Bm)
            If_vals[j] = ((H * d) + (g * Bm * area_ima) / (A_ar * u0)) / Nf
            fluxo_vals[j] = Nf * Bm * area_ima

        if np.isclose(x, h / 2, atol=1e-3) and Bg_h2 is None:
            Bg_h2 = Bm_vals * (area_ima / A_ar)
            If_h2 = If_vals.copy()

        idx_sort = np.argsort(If_vals)
        I_sorted = If_vals[idx_sort]
        phi_sorted = fluxo_vals[idx_sort]

        spline_phi = CubicSpline(I_sorted, phi_sorted)
        I0 = np.interp(0, Bm_vals, If_vals)
        I_range = np.linspace(I0, 0, Npts)
        phi_interp = spline_phi(I_range)
        coenergia[i] = trapezoid(phi_interp, I_range)

    spline_E = CubicSpline(posicoes[:-1], coenergia)
    dE_dx = spline_E.derivative()(posicoes[:-1])

    return Bg_h2, If_h2, dE_dx, posicoes[:-1]

# ===============================
# Execução principal
# ===============================
def main():
    plot_H_vs_B()
    Bg_r, If_r, F_r, pos_r = calcular_coenergia(interp_H_B_real, Nf_val)
    Bg_l, If_l, F_l, pos_l = calcular_coenergia(interp_H_B_lin, Nf_val)
    plot_fluxo_vs_corrente(If_r, Bg_r, If_l, Bg_l)
    plot_forca_vs_posicao(pos_r, F_r, pos_l, F_l)

if __name__ == "__main__":
    main()