import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score

# Caminho para o arquivo CSV
csv_file = '/path/to/diabetes_data.csv'

# Carregar os dados, descartando a primeira linha (cabeçalho)
data = pd.read_csv(csv_file, header=0)

# Separar as características (features) e a classe (label)
X = data.iloc[:, :-1].values  # Todas as colunas exceto a última
y = data.iloc[:, -1].values   # Última coluna

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir o modelo Deep MLP
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Fazer previsões
predictions = model.predict(X_test).round()

# Avaliação do modelo
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# Matriz de confusão e Índice Kappa
conf_matrix = confusion_matrix(y_test, predictions)
kappa_score = cohen_kappa_score(y_test, predictions)

print("Matriz de Confusão:")
print(conf_matrix)
print("Índice Kappa:", kappa_score)