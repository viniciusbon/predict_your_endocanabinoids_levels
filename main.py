import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Célula 2: Definição da Arquitetura do Modelo
# ESTA FUNÇÃO DEVE SER IDÊNTICA À USADA NO TREINAMENTO PARA RECONSTRUIR A ARQUITETURA CORRETAMENTE
def define_model(params, in_features, out_features):
    layers = []
    n_layers = params['n_layers']
    
    last_out_features = in_features
    for i in range(n_layers):
        n_units = params[f'n_units_l{i}']
        layers.append(nn.Linear(last_out_features, n_units))
        layers.append(nn.ReLU())
        dropout_rate = params[f'dropout_l{i}']
        layers.append(nn.Dropout(dropout_rate))
        last_out_features = n_units

    layers.append(nn.Linear(last_out_features, out_features))
    return nn.Sequential(*layers)

# Célula 3: Carregar os Artefatos Salvos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = 'melhor_modelo.pth'

# Carrega o checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Extrai os componentes
hyperparameters = checkpoint['hyperparameters']
scaler = checkpoint['scaler']
encoder = checkpoint['encoder']
input_features = scaler.mean_.shape[0]
output_features = len(encoder.classes_)

# Célula 4: Reconstruir o Modelo e Carregar os Pesos
# Recria a arquitetura do modelo com os hiperparâmetros salvos
modelo_carregado = define_model(hyperparameters, input_features, output_features).to(DEVICE)

# Carrega os pesos (o estado) do modelo treinado
modelo_carregado.load_state_dict(checkpoint['model_state_dict'])

# Define o modelo para o modo de avaliação (importante para camadas como Dropout e BatchNorm)
modelo_carregado.eval()

print("Modelo e artefatos carregados com sucesso!")
print("\nArquitetura do Modelo:")
print(modelo_carregado)

# Célula 5: Função para Fazer Predições em Novos Dados
def prever(dados_novos):
    """
    Recebe um DataFrame do Pandas com novos dados e retorna as predições.
    Os nomes das colunas devem ser os mesmos do dataset de treino.
    """
    # Garantir que os dados sejam um array numpy
    dados_array = np.array(dados_novos)

    # Aplicar o mesmo scaling usado no treino
    dados_scaled = scaler.transform(dados_array)

    # Converter para tensor
    dados_tensor = torch.tensor(dados_scaled, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        # Obter a saída do modelo (logits)
        outputs = modelo_carregado(dados_tensor)
        
        # Obter a classe prevista (índice com maior valor)
        _, predicted_indices = torch.max(outputs, 1)
        
        # Converter os índices de volta para os nomes das classes originais
        predicted_classes = encoder.inverse_transform(predicted_indices.cpu().numpy())
        
    return predicted_classes

# Célula 6: Exemplo de Uso da Função de Predição
# Crie um DataFrame com dados de exemplo (os nomes das colunas devem corresponder ao treino)
dados_exemplo = pd.DataFrame({
    'cnr1': [-0.02, 1.48, 0.73],
    'cnr2': [1.17, 0.83, 0.71],
    'dagl': [1.01, 1.05, 1.09],
    'magl': [0.87, 1.41, 0.69],
    'nape': [3.29, 4.10, 4.20],
    'faah': [1.06, 0.76, -0.14]
})

predicoes = prever(dados_exemplo)
print("\n--- Exemplo de Predição ---")
print("Dados de Entrada:")
print(dados_exemplo)
print("\nClasses Previstas:")
print(predicoes)


# Célula 7: (Opcional) Verificação da Performance no Conjunto de Teste Original
# Esta célula serve para validar que o modelo carregado tem o mesmo desempenho que o modelo ao final do treinamento.
print("\n--- Verificando a performance no conjunto de teste original ---")

# Carrega os dados originais para recriar o conjunto de teste
df_original = pd.read_excel('dados/Imputacao_mice_para_classificacao.xlsx')
X = df_original.drop('grupo', axis=1)
y_encoded_original = encoder.transform(df_original['grupo'])

_, X_test, _, y_test = train_test_split(
    X, y_encoded_original, test_size=0.3, random_state=42, stratify=y_encoded_original
)

# Fazer predições no conjunto de teste
y_pred_test = prever(X_test)
y_pred_test_encoded = encoder.transform(y_pred_test)


# Imprimir relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_test_encoded, target_names=encoder.classes_))

# Imprimir matriz de confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred_test_encoded)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Greens')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()