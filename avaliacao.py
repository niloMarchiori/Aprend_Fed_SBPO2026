import numpy as np
import pandas as pd
from FLPOPT.flopt import FLPOPT
from FLPOPT.flopt_util import avaliar_desempenho_nsgaii
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
T_max = 2**16

# ======================================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO ALGORITMO
# ======================================================================
instancia=FLPOPT(N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev)

columns=['T']
columns+=[f'f_{i}' for i in range(N)]
columns+=[f'beta_{i}' for i in range(N)]
columns+=[f'theta_{i}' for i in range(N)]
columns+=[f'psi_{i}' for i in range(N)]

T=T_max
t=0
NUM_RODADAS=2
dados_metricas=[{f'Rodada_{i}':None} for i in range(NUM_RODADAS)]
dados_solucao_escolhida=[{}]*NUM_RODADAS
while t<NUM_RODADAS:
    print(f'Rodada t={t}')
    df_cru, df_estatisticas, pf_empirica = avaliar_desempenho_nsgaii(
        instancia=instancia, 
        n_runs=30,       # Pelo menos 10 rodadas
        n_gen=200, 
        pop_size=150
    )

    dados_metricas[t]=df_estatisticas
    
    
    res = instancia.solve(n_gen=200, pop_size=150, seed=1)
    pesos = [0.3, 0.4, 0.3]
    idx= instancia.mcdm_pseudo_weights(pesos)
    solucao_vars=res.X[idx]
    dados_solucao_escolhida[t]=solucao_vars

    beta_t=np.array([solucao_vars[f'beta_{n}'] for n in range(N)])
    instancia.beta_h+=1-beta_t
    theta_t=np.array([solucao_vars[f'theta_{n}'] for n in range(N)])
    instancia.theta_prev=np.where(beta_t == 1, theta_t, instancia.theta_prev)
    t+=1
print(dados_metricas)
df_metricas = pd.concat(dados_metricas, keys=[f'Rodada_{i}' for i in range(NUM_RODADAS)])
df_metricas.to_csv('dados_metricas.csv')
df=pd.DataFrame(dados_solucao_escolhida)
df.to_csv('solucoes_teoricas.csv')