import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Binary
from pymoo.visualization.scatter import Scatter

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
# 2. DEFINIÇÃO DA CLASSE DO PROBLEMA
# ======================================================================
class FederatedLearningProblem(ElementwiseProblem):
    def __init__(self, N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev):
        self.N = N
        self.alpha = alpha
        self.c = c
        self.S = S
        self.f_min = f_min
        self.f_max = f_max
        self.epsilon_0 = epsilon_0
        self.theta_prev = theta_prev
        
        # Construindo o dicionário de Variáveis Mistas
        vars_dict = {}
        
        # Variável T (Contínua, Única global)
        vars_dict["T"] = Real(bounds=(T_min, T_max))
        
        for n in range(N):
            # f_n (Contínua)
            vars_dict[f"f_{n}"] = Real(bounds=(f_min[n], f_max[n]))
            # beta_n (Binária)
            vars_dict[f"beta_{n}"] = Binary()
            # psi_n (Inteira) - N*
            vars_dict[f"psi_{n}"] = Integer(bounds=(1, 100))
            # theta_n (Contínua) - Limitada em [0.01, 0.99] para evitar div/0 e log(0)
            vars_dict[f"theta_{n}"] = Real(bounds=(0.01, 0.99))
            
        super().__init__(
            vars=vars_dict,
            n_obj=3,          # AGORA TEMOS 3 OBJETIVOS
            n_ieq_constr=3*N  # TEMOS 3 RESTRIÇÕES PARA CADA 'n'
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Extração das variáveis
        T_val = x["T"]
        f_vals = np.array([x[f"f_{n}"] for n in range(self.N)])
        beta_vals = np.array([x[f"beta_{n}"] for n in range(self.N)])
        psi_vals = np.array([x[f"psi_{n}"] for n in range(self.N)])
        theta_vals = np.array([x[f"theta_{n}"] for n in range(self.N)])

        # Impor limites artificialmente no evaluate
        f_vals = np.clip(f_vals, self.f_min, self.f_max)
        psi_vals = np.maximum(psi_vals, 1)
        theta_vals = np.clip(theta_vals, 0.001, 0.999)

        # 2. Cálculo das Funções G(theta) e Psi(theta)
        # G(theta_n) = - log(1 - epsilon_0) / theta_n
        G_theta = -np.log2(1 - self.epsilon_0) / theta_vals
        
        # Psi(theta_n) = - log(1 - theta_n)
        Psi_theta = -np.log2(1 - theta_vals)

        # ====================================
        # FUNÇÕES OBJETIVO
        # ====================================
        # f1: min \sum (beta_n * psi_n * G(theta_n) * (alpha_n/2) * c_n * S_n * f_n^2)
        obj1 = np.sum(beta_vals * psi_vals * G_theta * (self.alpha / 2) * self.c * self.S * (f_vals**2))
        
        # f2: max \sum beta_n -> min -\sum beta_n
        obj2 = -np.sum(beta_vals)
        
        # f3: min G(theta_n) * T
        # Assumindo que queremos minimizar o tempo total ponderado pelos clientes selecionados
        # (Se a intenção for pegar o tempo máximo do pior cliente, use: np.max(beta_vals * G_theta) * T_val)
        obj3 = np.sum(beta_vals * G_theta * T_val)
        
        # ====================================
        # RESTRIÇÕES (g <= 0)
        # ====================================
        # g1: (psi_n * c_n * S_n / f_n) <= T  ==>  (psi_n * c_n * S_n / f_n) - T <= 0
        g1 = (psi_vals * self.c * self.S / f_vals) - T_val
        
        # g2: psi_n >= Psi(theta_n)  ==>  Psi(theta_n) - psi_n <= 0
        g2 = Psi_theta - psi_vals
        
        # g3: beta_n * theta_n >= beta_n * theta_n^{t-1}  ==>  beta_n * (theta_n^{t-1} - theta_n) <= 0
        g3 = beta_vals * (self.theta_prev - theta_vals)
        
        # O Pymoo exige que todas as restrições sejam passadas como uma lista/array 1D
        g_all = np.concatenate([g1, g2, g3])
        
        # Atribuindo saídas
        out["F"] = [obj1, obj2, obj3]
        out["G"] = g_all

# ======================================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO ALGORITMO
# ======================================================================
problem = FederatedLearningProblem(N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev)

# Instanciando operadores para variáveis mistas
dup_elim = MixedVariableDuplicateElimination()
mating_op = MixedVariableMating(eliminate_duplicates=dup_elim)

algorithm = NSGA2(
    pop_size=150,
    sampling=MixedVariableSampling(),
    mating=mating_op,
    eliminate_duplicates=dup_elim
)

print("Iniciando a otimização com 3 objetivos...")
res = minimize(
    problem,
    algorithm,
    seed=1,
    save_history=True,
    # verbose=True,
    termination=('n_gen', 500) # Mais gerações costumam ser necessárias para 3 objetivos
)

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



