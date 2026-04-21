import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Binary
from pymoo.visualization.scatter import Scatter
from FLPOPT.flopt import FLPOPT
# ======================================================================
# 1. DEFINIÇÃO DAS CONSTANTES DO PROBLEMA
# ======================================================================
N = 6
alpha = np.ones(N) * 2E-18
c = np.array([9,5,4,8,4,11])
S = np.array([32.6, 54.1, 81.2, 43.3, 65.0, 21.6]) * 1E6 # Substituiu o antigo D_n

# Limites de f_n
f_min = np.ones(N) * 1.3E9
f_max = np.array([2.3, 2.8, 2.7, 2.1, 2.5, 2.1]) * 1E9

# Novas constantes
epsilon_0 = 0.98                 # Precisão global desejada (exemplo)
theta_prev = np.ones(N) * 0.1    # Theta no tempo t-1 para a restrição de evolução

# Limites para a nova variável contínua T
T_min = 1.0
T_max = 500.0

# ======================================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO ALGORITMO
# ======================================================================
instancia=FLPOPT(N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev)

print("Iniciando a otimização com 3 objetivos...")
res = instancia.solve(n_gen=500, seed=1, verbose=True)

for solucao in res.F:
    solucao[1]=-solucao[1]

# ======================================================================
# 4. EXIBIÇÃO DOS RESULTADOS E PLOTAGEM
# ======================================================================
print("\n--- DETALHAMENTO DAS SOLUÇÕES DA FRONTEIRA DE PARETO ---")

if res.F is not None:
    # Mostrando os 5 primeiros indivíduos do Pareto final
    for i in range(len(res.X)):
        solucao_vars = res.X[i]
        objs = res.F[i]
        
        print(f"\n[Solução {i+1}]")
        print(f"Objetivos -> f1 (Energia): {objs[0]:.4e} | f2 (Beta): {objs[1]:.0f} | f3 (Tempo*Rodadas): {objs[2]:.4f}")
        print(f"Variável Global -> T = {solucao_vars['T']:.4f}")
        print("Variáveis Locais:")
        for n in range(N):
            val_f = solucao_vars[f"f_{n}"]
            val_beta = solucao_vars[f"beta_{n}"]
            val_psi = solucao_vars[f"psi_{n}"]
            val_theta = solucao_vars[f"theta_{n}"]
            print(f"  n={n}: f_{n} = {val_f:.4e} | beta_{n} = {val_beta} | psi_{n} = {val_psi} | PSI_{n}={-np.log2(1 - val_theta):.4f} | theta_{n} = {val_theta:.4f} | T_{n} = {val_psi*c[n]*S[n]/val_f:.4f}")
    plot = Scatter(title="Fronteira de Pareto (3 Objetivos)", angle=(45, 45))
    plot.add(res.F)
    plot.save("saida")
    plot.show()

            
else:
    print("Nenhuma solução viável foi encontrada. Considere aumentar T_max ou ajustar theta_prev.")



