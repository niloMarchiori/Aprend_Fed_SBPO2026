
import numpy as np
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

def print_solution_details(N,objs, solucao_vars,c,s):
    print(f"Objetivos -> f1 (Energia): {objs[0]:.4e} | f2 (Beta): {objs[1]:.2f} | f3 (Tempo*Rodadas): {objs[2]:.4f}")
    print(f"Variável Global -> T = {solucao_vars['T']:.4f}")
    print("Variáveis Locais:")
    for n in range(N):
        val_f = solucao_vars[f"f_{n}"]
        val_beta = solucao_vars[f"beta_{n}"]
        val_psi = solucao_vars[f"psi_{n}"]
        val_theta = solucao_vars[f"theta_{n}"]
        print(f"  n={n}: f_{n} = {val_f:.4e} | beta_{n} = {f'{val_beta} ' if val_beta else val_beta } | psi_{n} = {val_psi} | PSI_{n}={-np.log2(1 - val_theta):.4f} | theta_{n} = {val_theta:.4f} | T_{n} = {val_psi*c[n]*s[n]/val_f:.4f}")



def avaliar_desempenho_nsgaii(problema, n_runs=10, n_gen=200, pop_size=150):
    """
    Executa o NSGA-II múltiplas vezes e retorna a análise estatística de 
    Hipervolume (HV) e Inverted Generational Distance (IGD).
    """
    print(f"Iniciando a extração de métricas ({n_runs} execuções)...")
    
    # Configuração dos Operadores para Variáveis Mistas
    dup_elim = MixedVariableDuplicateElimination()
    mating_op = MixedVariableMating(eliminate_duplicates=dup_elim)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=mating_op,
        eliminate_duplicates=dup_elim
    )
    
    historico_F = [] # Armazenará o espaço de objetivos de todas as execuções
    
    # PASSO 1: Múltiplas Execuções
    for semente in range(n_runs):
        print(f" -> Rodada {semente+1}/{n_runs} (Seed: {semente})", end="\r")
        
        res = minimize(
            problema,
            algorithm,
            termination=('n_gen', n_gen),
            seed=semente,
            verbose=False
        )
        
        # Ignora casos onde o sistema falhou em convergir
        if res.F is not None:
            historico_F.append(res.F)
    
    print("\nExecuções finalizadas. Calculando fronteira empírica...")
    
    # PASSO 2: Criando a Fronteira de Pareto Empírica (Aproximação do Ótimo)
    # Empilhamos todos os indivíduos de todas as execuções
    todas_solucoes = np.vstack(historico_F)
    
    # Extraímos apenas as soluções não-dominadas deste grupão
    I = NonDominatedSorting().do(todas_solucoes, only_non_dominated_front=True)
    fronteira_empirica = todas_solucoes[I]
    
    # PASSO 3: Definindo o Ponto de Referência (Nadir) para o Hipervolume
    # Pega o pior caso de cada objetivo na fronteira e adiciona 5% de margem
    nadir_point = fronteira_empirica.max(axis=0) * 1.05 
    
    # Inicializando as calculadoras de métricas do pymoo
    ind_hv = Hypervolume(ref_point=nadir_point)
    ind_igd = IGD(fronteira_empirica)
    
    # PASSO 4: Calculando as métricas para CADA execução individual
    resultados_metricas = []
    
    for F_rodada in historico_F:
        # Calcula HV
        hv_val = ind_hv.do(F_rodada)
        
        # Calcula IGD
        igd_val = ind_igd.do(F_rodada)
        
        resultados_metricas.append({
            "Hypervolume": hv_val,
            "IGD": igd_val
        })
        
    df_metricas = pd.DataFrame(resultados_metricas)
    
    # PASSO 5: Consolidando Média e Desvio Padrão
    estatisticas = pd.DataFrame({
        "Média": df_metricas.mean(),
        "Desvio Padrão": df_metricas.std(),
        "Mínimo": df_metricas.min(),
        "Máximo": df_metricas.max()
    })
    
    return df_metricas, estatisticas, fronteira_empirica
