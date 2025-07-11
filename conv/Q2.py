from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import sympy as sp_sim
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.misc import derivative
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm
from scipy.integrate import quad

# Configurações de estilo e fonte
plt.style.use('ggplot')
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Constantes
pi = np.pi
permeabilidade = pi * (4e-7)  # H/m

# Dados do material
h_nucleo = np.array([0, 68, 135, 203, 271, 338, 406, 474, 542, 609, 1100, 1500, 2500, 4000, 5000, 9000, 12000, 20000, 25000])  # A/m
b_nucleo = np.array([0, 0.733, 1.205, 1.424, 1.517, 1.560, 1.588, 1.617, 1.631, 1.646, 1.689, 1.703, 1.724, 1.731, 1.738, 1.761, 1.770, 1.800, 1.816])  # T

# Geometria e parâmetros do sistema
compr_gap = 0.45e-3         # m
raio_rotor = 6.3e-2         # m
compr_sistema = 8e-2        # m
num_voltas = 90             # número de voltas
compr_nucleo = 76e-2        # m    (Considerando maior do que 50e-2)
angulo_rotor = 30           # grau

num_pontos = 1000

# Interpolação da curva B x H
interp_b = sp.interpolate.interp1d(h_nucleo, b_nucleo, kind="cubic")
h_valores = np.linspace(0, 25000, num_pontos)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(h_valores, interp_b(h_valores), lw=2)
ax1.set_xlabel("H [A/m]")
ax1.set_ylabel("B [T]")
ax1.set_title("Curva B x H")
ax1.grid(True)
fig1.tight_layout()
plt.show()

# Função para calcular a área efetiva
def calcula_area(angulo_graus):
    # Cálculo da área com base no ângulo em graus
    area_efetiva = compr_sistema * (np.deg2rad(30 - abs(angulo_graus)) * raio_rotor)
    return area_efetiva

A_const = calcula_area(0)
conjunto_angulos = np.array([-29.9, -20, -10, 0, 10, 20, 29.9])

fig2, ax2 = plt.subplots(figsize=(8, 4))
fig3, ax3 = plt.subplots(figsize=(8, 4))

for ang in conjunto_angulos:
    area_atual = calcula_area(ang)
    if area_atual != 0:
        fluxo_concentrado = num_voltas * b_nucleo * A_const
        b_gap = (b_nucleo * A_const) / area_atual
        corrente_real = (h_nucleo * compr_nucleo + (b_gap * 2 * compr_gap) / permeabilidade) / num_voltas
        corrente_ideal = np.ones(len(h_nucleo)) * ((b_gap * 2 * compr_gap) / permeabilidade) / num_voltas

        ax2.plot(corrente_real, fluxo_concentrado, lw=2, label=rf"$\theta = {ang}°$")
        ax3.plot(corrente_ideal, fluxo_concentrado, lw=2, label=rf"$\theta = {ang}°$")

estilos_linha = ['-', '--', '-.', '-', ':', '-.', '-.']
cores = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']

# Aplicar os estilos e cores aos gráficos do ax2
for i, linha in enumerate(ax2.get_lines()):
    linha.set_linestyle(estilos_linha[i % len(estilos_linha)])
    linha.set_color(cores[i % len(cores)])

# Aplicar os estilos e cores aos gráficos do ax3
for i, linha in enumerate(ax3.get_lines()):
    linha.set_linestyle(estilos_linha[i % len(estilos_linha)])
    linha.set_color(cores[i % len(cores)])

ax2.set_xlim(0, 50)
ax3.set_xlim(0, 50)
ax2.set_title(r"$\lambda \times I$ (Real)")
ax2.set_xlabel(r"$I\ [A]$")
ax2.set_ylabel(r"$\lambda\ [Wb.e]$")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()

ax3.set_title(r"$\lambda \times I$ (Ideal)")
ax3.set_xlabel(r"$I\ [A]$")
ax3.set_ylabel(r"$\lambda\ [Wb.e]$")
ax3.legend()
ax3.grid(True)
fig3.tight_layout()

plt.show()

# Cálculo das correntes para uma condição específica
b_gap_const = (1.8 * A_const) / calcula_area(0)
corrente_1_8_real = (20000 * compr_nucleo + (b_gap_const * 2 * compr_gap) / permeabilidade) / num_voltas
corrente_1_8_ideal = ((b_gap_const * 2 * compr_gap) / permeabilidade) / num_voltas
print(f"Corrente Real: I = {corrente_1_8_real:.2f}A")
print(f"Corrente Ideal: I = {corrente_1_8_ideal:.2f}A")

# Variação de ângulo para análise de energia e torque
theta_valores = np.linspace(-29.9, 29.9, num_pontos)
energia_real = []
energia_ideal = []

for ang in theta_valores:
    fluxo_conc2 = num_voltas * b_nucleo * A_const
    b_gap_ang = (b_nucleo * A_const) / calcula_area(ang)
    corrente_real2 = (h_nucleo * compr_nucleo + (b_gap_ang * 2 * compr_gap) / permeabilidade) / num_voltas
    corrente_ideal2 = np.ones(len(h_nucleo)) * ((b_gap_ang * 2 * compr_gap) / permeabilidade) / num_voltas

    corrente_enrolamento = np.linspace(0, corrente_1_8_real, num_pontos)

    interp_real = sp.interpolate.interp1d(corrente_real2, fluxo_conc2, kind="cubic", fill_value="extrapolate")
    interp_ideal = sp.interpolate.interp1d(corrente_ideal2, fluxo_conc2, kind="cubic", fill_value="extrapolate")

    energia_real = np.append(energia_real, sp.integrate.trapezoid(interp_real(corrente_enrolamento), corrente_enrolamento))
    energia_ideal = np.append(energia_ideal, sp.integrate.trapezoid(interp_ideal(corrente_enrolamento), corrente_enrolamento))

torque_ideal = np.gradient(energia_ideal, theta_valores)
torque_real = np.gradient(energia_real, theta_valores)

fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(12, 4))
ax4_1.plot(theta_valores, energia_ideal, lw=2)
ax4_1.set_title("Energia por Posição (Ideal)")
ax4_1.set_xlabel(r"$\theta\ [°]$")
ax4_1.set_ylabel("Energia [J]")
ax4_1.grid(True)
ax4_2.plot(theta_valores, torque_ideal, lw=2)
ax4_2.set_title("Torque por Posição (Ideal)")
ax4_2.set_xlabel(r"$\theta\ [°]$")
ax4_2.set_ylabel("Torque [N.m]")
ax4_2.grid(True)
fig4.tight_layout()
plt.show()

fig5, (ax5_1, ax5_2) = plt.subplots(1, 2, figsize=(12, 4))
ax5_1.plot(theta_valores, energia_real, lw=2)
ax5_1.set_title("Energia por Posição (Real)")
ax5_1.set_xlabel(r"$\theta\ [°]$")
ax5_1.set_ylabel("Energia [J]")
ax5_1.grid(True)
ax5_2.plot(theta_valores, torque_real, lw=2)
ax5_2.set_title("Torque por Posição (Real)")
ax5_2.set_xlabel(r"$\theta\ [°]$")
ax5_2.set_ylabel("Torque [N.m]")
ax5_2.grid(True)
fig5.tight_layout()
plt.show()


# Razão de potências na condição ideal (independente do ângulo)
fluxo_conc3 = num_voltas * b_nucleo * A_const
b_gap_10 = (b_nucleo * A_const) / calcula_area(10)
corrente_real3 = (h_nucleo * compr_nucleo + (b_gap_10 * 2 * compr_gap) / permeabilidade) / num_voltas
corrente_ideal3 = np.ones(len(h_nucleo)) * ((b_gap_10 * 2 * compr_gap) / permeabilidade) / num_voltas

fig6, ax6 = plt.subplots(figsize=(8, 4))
ax6.plot(corrente_ideal3, fluxo_conc3, lw=2, label=r"$\theta = 10°$")
ax6.set_title("Situação Ideal")
ax6.set_xlabel(r"$I\ [A]$")
ax6.set_ylabel(r"$\lambda\ [Wb.e]$")
ax6.legend()
ax6.grid(True)
fig6.tight_layout()
plt.show()


# Cálculo da razão de potências na condição real para diferentes ângulos
lista_angulos = [0, 5, 10, 15, 20, 25, 29.9]
fluxo_conc = num_voltas * b_nucleo * A_const

for ang in lista_angulos:
    area_gap = calcula_area(ang)
    b_gap = (b_nucleo * A_const) / area_gap

    # Corrente real considerando o campo do núcleo e o gap
    corrente_real = (h_nucleo * compr_nucleo + (b_gap * 2 * compr_gap) / permeabilidade) / num_voltas

    # Corrente ideal para o ângulo (valor constante)
    corrente_ideal_valor = ((b_gap * 2 * compr_gap) / permeabilidade) / num_voltas
    corrente_ideal = np.full_like(b_nucleo, corrente_ideal_valor)

    # Integração numérica utilizando np.trapz para obter as áreas
    area_superior = np.trapz(fluxo_conc, corrente_real)
    area_inferior = np.trapz(corrente_real, fluxo_conc)

    # Potência total aplicada (AC) e potência efetiva no motor
    potencia_ac = area_superior + area_inferior
    potencia_motor = area_superior

    razao_potencia = potencia_motor / potencia_ac
    print(f"Para theta = {ang}: Razão Real = {razao_potencia:.5f}")