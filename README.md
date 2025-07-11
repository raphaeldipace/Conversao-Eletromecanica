# Conversão Eletromecânica

# Autor: Raphael Di Pace

# 🧲 Análise de Força Magnética em Sistemas com Ímã Permanente

Este repositório contém uma simulação numérica de um sistema magnético com **ímã permanente**, considerando diferentes modelos da curva B-H e analisando:

- A **força magnética** gerada em função da posição do êmbolo.
- A **densidade de fluxo no entreferro**.
- A **curva de magnetização B-H** do material magnético.
- A **coenergia magnética** no circuito.

---

## ⚙️ Funcionalidades

- Modelagem da curva B-H com base em dados reais e linearizados.
- Cálculo da **coenergia magnética** por integração numérica.
- Estimativa da força magnética a partir da derivada da coenergia.
- Análise gráfica detalhada:
  - Curva B-H (real e linear).
  - Densidade de fluxo no entreferro vs corrente.
  - Força magnética vs posição.

---

## 🔍 Principais Modelos Avaliados

- **Modelo Real:** Considera a não linearidade da curva B-H.
- **Modelo Linear:** Usa aproximação linear com a permeabilidade relativa constante.

---

## ▶️ Como Executar

1. Instale as bibliotecas necessárias:

```bash
pip install numpy matplotlib scipy