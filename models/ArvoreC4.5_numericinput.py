import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

# Ler o arquivo CSV com as colunas na primeira linha
df = pd.read_csv('endereco do arquivo')

# Identificar automaticamente as características e a classe
features = list(df.columns[:-1])
target = df.columns[-1] 

# Separar os dados em características (X) e rótulos (y)
X = df[features]
y = df[target]

# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de árvore de decisão C4.5
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Treinar o modelo usando os dados de treinamento
clf.fit(X_train, y_train)

# Testar o modelo usando os dados de teste
accuracy = clf.score(X_test, y_test)
print("Acurácia:", accuracy)

# Imprimir as regras da árvore de decisão
tree_rules = export_text(clf, feature_names=features)
print(tree_rules)
