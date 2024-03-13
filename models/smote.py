from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# Ler o arquivo CSV com as colunas na primeira linha
df = pd.read_csv('../data/diabetes_prediction_dataset.csv')

# Gerando um conjunto de dados desbalanceado para exemplo
X, y = make_classification(n_classes=2, class_sep=2,
                          weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                          n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

print('Distribuição original das classes:', Counter(y))

# Dividindo em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Aplicando SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print('Distribuição das classes após o SMOTE:', Counter(y_res))

# Visualizando a distribuição das classes antes e depois do SMOTE
plt.figure(figsize=(12, 5))

plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Diabéticos",s=30)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Não diabéticos",s=30)
plt.title("Base de dados tradicional")
plt.legend()

plt.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label="Diabéticos",s=30)
plt.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label="Não diabéticos",s=30)
plt.title("Base de dados processada pelo SMOTE")
plt.legend()

plt.show()