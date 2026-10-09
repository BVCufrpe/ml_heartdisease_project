import pandas as pd

# Primeiro ponto: definição das variáveis com base no dicionário do dataset - https://archive.ics.uci.edu/dataset/45/heart+disease

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

df = pd.read_csv(url, names=cols)

df['num_target'] = df['num_target'].apply(lambda x: 1 if x>0 else 0)

target_var = 'num_target'
linking_vars = df.columns.drop('num_target').tolist()

print("Teste v1:")
print(f"formato dos dados: {df.shape[0]} linhas e {df.shape[1]} colunas. \n")

print("Distribuição das classes no alvo (Balanceamento): ")
distribution = df['num_target'].value_counts(normalize=True) * 100
print(f"0 (Sem doença): {distribution[0]:.2f}%")
print(f"1 (Com doença): {distribution[1]:.2f}%")
