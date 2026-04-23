import numpy as np
from FLPOPT.flopt import FLPOPT
from FLPOPT.flopt_util import print_solution_details

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
res = instancia.solve(n_gen=200, pop_size=100, seed=1)

# ======================================================================
# 4. EXIBIÇÃO DOS RESULTADOS E PLOTAGEM
# ======================================================================
print("\n--- DETALHAMENTO DAS SOLUÇÕES DA FRONTEIRA DE PARETO ---")

if res.F is not None:
    # for i in range(len(res.X)):
    #     solucao_vars = res.X[i]
    #     objs = res.F[i]
    #     print_solution_details(N, objs, solucao_vars, c, S)

    #--- SOLUÇÃO SELECIONADA PELO MÉTODO DE MCDM (PSEUDO PESOS) ---#
    pesos = [0.3, 0.4, 0.3]  # Exemplo de pesos para os objetivos
    idx= instancia.mcdm_pseudo_weights(pesos,verbose=True)

    #--- SOLUÇÃO SELECIONADA PELO MÉTODO DE MCDM (High Tradeoff Points) ---#
    # idx_knee = instancia.mcdm_knee_point(verbose=True)
    
    
    
    # instancia.scatterplot()
    
            
else:
    print("Nenhuma solução viável foi encontrada. Considere aumentar T_max ou ajustar theta_prev.")