import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    'font.size': 20,           # Tamanho base da fonte para tudo
    'legend.fontsize': 18,     # Tamanho específico da legenda
    'axes.labelsize': 20,      # Tamanho dos rótulos X e Y
    'axes.titlesize': 20,      # Tamanho do título do gráfico
    'xtick.labelsize': 14,     # Tamanho dos números no eixo X
    'ytick.labelsize': 14,     # Tamanho dos números no eixo Y
    'figure.figsize': (10, 6)  # Tamanho padrão da figura
})

N = 6
alpha = np.ones(N) * 2E-28
c = np.array([9,5,4,8,4,11])
S = np.array([32.6, 54.1, 81.2, 43.3, 65.0, 21.6]) * 1E6 # Substituiu o antigo D_n

df_evo_teorica=pd.read_csv('Analise/solucoes_teoricas.csv')


df_evo_teorica.head()
df_consumos_individuais=pd.DataFrame()
for i in range(N):
    f_i=df_evo_teorica[f'f_{i}']
    beta_i=df_evo_teorica[f'beta_{i}']
    psi_i=df_evo_teorica[f'psi_{i}']
    df_consumos_individuais[f'consumo_sta{i}']=beta_i*psi_i*alpha[i]*c[i]*S[i]*(f_i.apply(lambda x: (x)**2))/2
    df_consumos_individuais[f'consumo_sta{i}']=df_consumos_individuais[f'consumo_sta{i}']*0.3


df_consumos_individuais['consumo_total']=df_consumos_individuais.sum(axis=1)
df_evo_teorica['consumo_acumulado']=df_consumos_individuais['consumo_total'].cumsum()

df_evo_teorica['mean_acc']=df_evo_teorica[[f'theta_{i}' for i in range(N)]].mean(axis=1)
print(df_evo_teorica)

fig,ax=plt.subplots()
ax.plot(df_evo_teorica['consumo_acumulado'])
plt.show()

fig,ax=plt.subplots()
ax.plot(df_evo_teorica['mean_acc'])
plt.show()

fig,ax=plt.subplots()
ax.plot(df_evo_teorica['T'].cumsum(),df_evo_teorica['mean_acc'])
plt.show()
