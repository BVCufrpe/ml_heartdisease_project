import pandas as pd

# Primeiro ponto: definição das variáveis com base no dicionário do dataset - https://archive.ics.uci.edu/dataset/45/heart+disease
# na etapa IDA, é preciso verificar os tipos de cara variável, esse método faz parte da proposta da Exp. DA.
cols = [
    "age",
    "sex",
    "cp",
    "tresbps",
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
