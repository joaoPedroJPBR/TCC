# Projeto de Análise de Dados - Diabetes Prediction

## Descrição do Projeto

Este projeto envolve a análise de dados para previsão de diabetes, utilizando técnicas de aprendizado de máquina. O conjunto de dados utilizado é proveniente de [insira a fonte dos dados] e contém informações relevantes sobre pacientes, incluindo características como idade, histórico de tabagismo, índice de massa corporal (IMC), nível de HbA1c e outros fatores.

## Conteúdo do Repositório

1. **datasets/**: Pasta contendo o conjunto de dados CSV utilizado no projeto.
   - `diabetes_prediction_dataset.csv`: O conjunto de dados original.

2. **notebooks/**: Pasta contendo Jupyter Notebooks relacionados à análise de dados.
   - `data_exploration.ipynb`: Notebook para explorar e visualizar o conjunto de dados.
   - `model_training.ipynb`: Notebook para treinar e avaliar modelos de aprendizado de máquina.

3. **src/**: Pasta com scripts Python utilizados no projeto.
   - `preprocess.py`: Script para pré-processamento de dados.
   - `model.py`: Script para definição e treinamento de modelos.

4. **docs/**: Documentação do projeto.
   - `README.md`: Este arquivo.

## Etapas do Projeto

### 1. Exploração de Dados

- Utilizamos o notebook `data_exploration.ipynb` para explorar e visualizar o conjunto de dados.
- Removemos as colunas desnecessárias, como "gender" e "smoking_history".

### 2. Pré-Processamento

- Realizamos o pré-processamento de dados no script `preprocess.py`.
- Separamos os dados em características (X) e rótulos (y).
- Dividimos o conjunto de dados em conjuntos de treinamento e teste.

### 3. Treinamento do Modelo

- Utilizamos um modelo de Árvore de Decisão C4.5 para prever a ocorrência de diabetes.
- O treinamento e avaliação do modelo são realizados no notebook `model_training.ipynb`.

### 4. Avaliação do Modelo

- Calculamos a acurácia e o índice Kappa para avaliar o desempenho do modelo.

## Como Executar

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/diabetes-prediction.git
cd diabetes-prediction
