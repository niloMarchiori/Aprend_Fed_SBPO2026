import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Choice, Binary


# Definindo constantes do problema (Exemplo: N=5)
N = 6
alpha = np.ones(N)*2E-18
c = np.array([9,5,4,8,4,11])
D = np.array([32.6,54.1,81.2,43.3,65.,21.6])*1E6
T = 6
f_min = np.ones(N) * 1.3E9
f_max = np.array([2.3,2.8,2.7,2.1,2.5,2.1])*1E9

class OtimizacaoRecursos(ElementwiseProblem):
    def __init__(self, N, alpha, c, D, T, f_min, f_max):
        self.N = N
        self.alpha = alpha
        self.c = c
        self.D = D
        self.T = T
        
        # Correção: Utilizar as classes de Variáveis do Pymoo para definir tipo e limites
        vars_dict = {}
        for n in range(N):
            # f_n é contínuo (real). Os limites f_min e f_max já podem ser passados aqui
            vars_dict[f"f_{n}"] = Real(bounds=(f_min[n], f_max[n]))
            
            # beta_n é binário
            vars_dict[f"beta_{n}"] = Binary()
            
            # psi_n é inteiro (em N*). 
            # O Pymoo exige um limite superior para amostrar variáveis inteiras. 
            # Abaixo o limite máximo foi arbitrado em 100 (ajuste de acordo com a sua aplicação)
            vars_dict[f"psi_{n}"] = Integer(bounds=(1, 100)) 
            
        super().__init__(
            vars=vars_dict,
            n_obj=2,       
            n_ieq_constr=N 
        )
        
        self.f_min = f_min
        self.f_max = f_max

    def _evaluate(self, x, out, *args, **kwargs):
        # Extrair variáveis do dicionário `x`
        f_vals = np.array([x[f"f_{n}"] for n in range(self.N)])
        beta_vals = np.array([x[f"beta_{n}"] for n in range(self.N)])
        psi_vals = np.array([x[f"psi_{n}"] for n in range(self.N)])

        # Impor artificialmente limites das variáveis no evaluate (garantia extra)
        f_vals = np.clip(f_vals, self.f_min, self.f_max)
        psi_vals = np.maximum(psi_vals, 1) # psi_n in N^* (maior ou igual a 1)

        # ------------------------------------
        # Função Objetivo 1 (Minimização)
        # ------------------------------------
        obj1 = np.sum(self.alpha * self.c * (f_vals**2) * beta_vals * psi_vals)
        
        # ------------------------------------
        # Função Objetivo 2 (Convertida para Minimização)
        # ------------------------------------
        obj2 = -np.sum(beta_vals) # Multiplicado por -1 [1]
        
        # ------------------------------------
        # Restrições de Desigualdade (<= 0)
        # ------------------------------------
        # g_n: (\psi_n * c_n * D_n / f_n) - T <= 0
        g = (psi_vals * self.c * self.D / f_vals) - self.T
        
        # Atribuindo objetivos e restrições aos vetores de saída [4]
        out["F"] = [obj1, obj2]
        out["G"] = g

# Instanciando o Problema
problem = OtimizacaoRecursos(N, alpha, c, D, T, f_min, f_max)

# 1. Instanciar o operador de eliminação de duplicatas para variáveis mistas
dup_elim = MixedVariableDuplicateElimination()

# 2. Instanciar o operador de cruzamento repassando o eliminador de duplicatas
mating_op = MixedVariableMating(eliminate_duplicates=dup_elim)

# 3. Configurar o Algoritmo NSGA-II
algorithm = NSGA2(
    pop_size=100,
    sampling=MixedVariableSampling(),
    mating=mating_op,
    eliminate_duplicates=dup_elim
)

# 4. Executando a Otimização
res = minimize(
    problem,
    algorithm,
    # seed=1,
    save_history=True,
    # verbose=True,
    termination=('n_gen', 200) # Condição de parada: limite de gerações [7, 8]
)

# Os resultados pertencentes à Fronteira de Pareto otimizada estarão contidos em 'res' [9].
print("Objetivos das melhores soluções encontradas:")
print(res.F)

# Acessando as variáveis de decisão (res.X) e as funções objetivo (res.F)
solucoes = res.X
objetivos = res.F

print("\n--- DETALHAMENTO DAS SOLUÇÕES DA FRONTEIRA DE PARETO ---")

if solucoes is not None:
    for i in range(min(5, len(solucoes))):
        solucao_vars = solucoes[i]
        objs = objetivos[i]
        
        print(f"\n[Solução {i+1}]")
        # CORREÇÃO: Adicionado  após objs
        print(f"Objetivos -> f1 (Custo): {objs[0]:.4f} | f2 (Beta agrupado): {objs[1]:.4f}")
        print("Variáveis de Escolha:")
        
        for n in range(N):
            val_f = solucao_vars[f"f_{n}"]
            val_beta = solucao_vars[f"beta_{n}"]
            val_psi = solucao_vars[f"psi_{n}"]
            
            print(f"  n={n}: f_{n} = {val_f:.4f} | beta_{n} = {val_beta} | psi_{n} = {val_psi}")

else:
    print("Nenhuma solução viável foi encontrada pelo algoritmo.")

from pymoo.visualization.scatter import Scatter

# Extraindo a função objetivo (F) de toda a população em TODAS as gerações
# res.history contém o estado do algoritmo em cada geração [1]
historico_F = [algo.pop.get("F") for algo in res.history]

# Empilhando todas as gerações em um único array 2D
todos_pontos = np.vstack(historico_F)

mascara = todos_pontos[:, 0] <= 1250

# O array 'pontos_filtrados' agora conterá apenas as linhas onde a condição foi Verdadeira
pontos_filtrados = todos_pontos[mascara]

# Instanciando o gráfico
plot = Scatter(title="Todas as Soluções Avaliadas vs Fronteira de Pareto", legend=True)

# 1º Adicionamos todos os pontos explorados pelo algoritmo (em cinza e levemente transparentes) [2]
plot.add(pontos_filtrados, color="lightgrey", alpha=0.3, label="Todas as avaliações")

# 2º Adicionamos as soluções da Fronteira de Pareto por cima (em vermelho) [2]
if res.F is not None:
    plot.add(res.F, color="red", marker="o", s=30, label="Fronteira de Pareto")

# Exibindo o gráfico
plot.show()
