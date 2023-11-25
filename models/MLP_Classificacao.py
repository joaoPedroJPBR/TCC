import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Ler o arquivo .csv
data = pd.read_csv("seuarquivo.csv")

# 2. Converta todos os atributos não numéricos em numéricos
le = preprocessing.LabelEncoder()
for column in data.columns:
    if data[column].dtype == type(object):
        data[column] = le.fit_transform(data[column])

# 3. Separe as entradas (X) e a classe (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 4. Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Treine a MLP
clf = MLPClassifier(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 6. Faça as previsões
y_pred = clf.predict(X_test)

# 7. Imprima a acurácia e a matriz de confusão
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
