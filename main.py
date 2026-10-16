import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Primeiro ponto: definição das variáveis com base no dicionário do dataset - https://archive.ics.uci.edu/dataset/45/heart+disease
# na etapa IDA, é preciso verificar os tipos de cara variável, esse método faz parte da proposta da Exp. DA.
cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num_target",
        ]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Quand o pandas importa um dataset, ele "tenta" inferir os tipos de dados de cada parâmetro (valor). Geralmente
# os tipos dão "match" com a NumPy, como int64, float64 e etc. Quando existem tipos "misturados", temos o tipo object.
# Segundo a própria doc do pandas, existe um parâmetro (dtype) que pode ser passado nas funções 12
df = pd.read_csv(url, names=cols)
## *importante: as vezes, principalmente em datasets de pacientes reais, os dados podem vir nulos, como a scikit
## não lida bem com esses valores, é importante fazer uma imputação de valores passíveis de tratamento.

df['num_target'] = df['num_target'].apply(lambda x: 1 if x>0 else 0)

target_var = 'num_target'
linking_vars = df.columns.drop('num_target').tolist()

print("Teste v1:")
print(f"formato dos dados: {df.shape[0]} linhas e {df.shape[1]} colunas. \n")

print("Distribuição das classes no alvo (Balanceamento): ")
distribution = df['num_target'].value_counts(normalize=True) * 100
print(f"0 (Sem doença): {distribution[0]:.2f}%")
print(f"1 (Com doença): {distribution[1]:.2f}%")


## A partir de agora, vamos para a EDA de fato:

print(df.dtypes) # vamos verificar todos os tipos dos dados.

print(df.isnull().sum()) # Esse daqui vai somar todos os valores que vieram null por cada coluna de dados.

text_cols = df.select_dtypes(include=['str']).columns # Identifica quais colunas vieram como "texto"

if len(text_cols) > 0:
    for col in text_cols:
        print(f"Valores únicos em {col}:\n {df[col].unique()} \n")

else: 
    print("Não foram encontrados textos ou tipos diferentes nas colunas numéricas")

# describe() -> contagem, média, desvio padrão, mínimo, quartis e máximo.
# ignora cols de texto (por padrão).

statistics = df.describe()

# as Rows viram os params e as cols viram os valores de média, std deviation, min, max & etc.
print(statistics.T)

# Aqui vamos começar a destrinchar os dados em gráficos, com o objetivo e verificar o balanceamento dos dados.

sb.set_theme(style="whitegrid")
plt.figure(figsize=(6,4))

sb.countplot(data=df, x='num_target', palette=['#1f77b4', '#d62728'])

plt.title("Class distribution - cardiac disease")
plt.xlabel("Diagnosis: (0 = Without Cardiac Disease, 1 = With Cardiac Disease)")
plt.ylabel("Number of patients")

file_name = 'target_distribution.png'
plt.savefig(file_name, dpi=600, bbox_inches='tight')
plt.close()

# Começando a análise da correlação de Pearson
plt.figure(figsize=(12,8))

correlation_matrix = df.corr(numeric_only=True) # ignora os casos onde o casting do pandas pra object acontecem (ca e thal)

sb.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1) # O heatmap é um ótimo tipo de gráfico para demonstrar a itensidade da correlação entre duas variáveis ([-1, 1]
plt.title("Correlação de Pearson (Heatmap)")

file_name_correlation = "mapa_correlacao.png"
plt.savefig(file_name_correlation, dpi=600, bbox_inches='tight')
plt.close()

## Histograma (tratando variáveis contínuas como discretas em faixas (bins)) da idade x variável alvo (ausência/presença da doença cardíaca)
plt.figure(figsize=(8,5))
sb.histplot(data=df, x='age', hue='num_target', kde=True, palette=['#1f77b4','#d62728'])
plt.title("Distribuição da Idade por Diagnóstico")
plt.xlabel("Idade (Anos)")
plt.ylabel("Frequência")

file_name_histogram = "histograma_idade_frequência.png"
plt.savefig(file_name_histogram, dpi=600, bbox_inches='tight')
plt.close()

## Box Plots para percentis (verificação de outliers, e validação dos mesmos), serão dois (colesterol e pressão arterial)

fig, axis = plt.subplots(1,2, figsize=(12,5))

sb.boxplot(data=df, x='num_target', y='trestbps', ax=axis[0], palette=['#1f77b4', '#d62728'])
axis[0].set_title('Pressão arterial em repouso (trestbps) vs Alvo (presença de doença cardíaca)')
axis[0].set_xlabel('Alvo: 0 - Sem doença | 1 - Com doença')
axis[0].set_ylabel('Pressão Arterial (mmHg)')

sb.boxplot(data=df, x='num_target', y='chol', ax=axis[1], palette=['#1f77b4','#d62728'])
axis[1].set_title('Colesterol vs Alvo')
axis[1].set_xlabel('Alvo: 0 - Sem doença | 1 - Com doença')
axis[1].set_ylabel('Colesterol (mg/dl)')

box_name = 'boxplots_outliers.png'
plt.savefig(box_name, dpi=600, bbox_inches='tight')
plt.close()
## Aqui eu começo o pré-processamento dos dados:

df.replace('?', np.nan, inplace=True)
# dando replace nos '?' por NaN (np.nan) e castando float nelas
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)

df = df.dropna()

## A partir daqui vamos começar de fato o pré-processamento, o scikit precisa que eu entregue os dados em duas matrizes
## distintas: a matriz das variáveis independentes e a matriz-coluna da variável alvo (num_target)

## Train-test split approach: conceito simples para evitar overfitting (o modelo "decora" as respostas e não analisa
## corretamente as correlações biológicas) >> fornecer uma qtd. limitada (e.g. 70%) dos dados para o modelo e guardar
## o restante para uma prova cega (Teste).

## Feature Scalling >> Conceito bem relevante: algumas variáveis variam em ordens de grandeza muito distintas entre 
## si (exemplo: Colesterol (centenas) vs nivelamento do segmento ST do eletrocardiograma (oldpeak) que varia em decimais
## Para isso é importante que ocorra um "nivelamento" (Normalização), aqui vamos usar o modelo Z-escore ( x - media / std )

## Para evitar o overfitting no nivelamento, vamos "treinar" o modelo apenas com os dados de treino (os 70%)

independent_vars = df.drop("num_target", axis=1)
dependent_var = df["num_target"]

#stratify mantém a proporção de pacientes com/sem doença
independent_vars_train, independent_vars_test, dependent_var_train, dependent_var_test = train_test_split(independent_vars, dependent_var, test_size=0.3, random_state=42, stratify=dependent_var)

scaler = StandardScaler()

ind_vars_train_scaled = scaler.fit_transform(independent_vars_train)
ind_vars_test_scaled = scaler.fit_transform(independent_vars_test)

print(f"Dados de treino: {ind_vars_train_scaled.shape[0]} pacientes")
print(f"Dados de teste: {ind_vars_test_scaled.shape[0]} pacientes")

