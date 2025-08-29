# Predição de Endocanabinoides com Redes Neurais a partir de Composição Corporal e Perfil de Alimentação

Este projeto implementa uma **rede neural usando PyTorch** para realizar **predições de biomarcadores (endocanabinoides)** a partir de dados simulados de saúde e composição corporal.  

O modelo utiliza hiperparâmetros previamente otimizados e carrega pesos de treinamento salvos, permitindo executar **inferência em novos dados simulados**.

## 🚀 Você pode colocar os SUAS medidas para serem avaliadas!

 ⚠️ ATENÇÃO ⚠️ Apesar de ser proveniente de um estudo clínico, este projeto foi desenvolvido **exclusivamente para fins acadêmicos**.  
 Não deve ser utilizado para diagnósticos médicos ou aplicações clínicas reais.


## 🚀 Funcionalidades

- Definição da arquitetura da rede neural a partir dos hiperparâmetros otimizados.  
- Geração de dados simulados de variáveis como:
Idade, peso, IMC, massa muscular, massa gorda, ingestão calórica e macronutrientes, etc.  
- Carregamento de pesos e escaladores salvos 
- Predição de biomarcadores:  
  - `cnr1 - Receptor Canabinoide tipo 1 (CB1)`  
  - `cnr2 - Receptor Canabinoide tipo 2 (CB2)`  
  - `dagl - Enzima de Síntese de 2-AG`  
  - `magl - Enzima de Degradação de 2-A`  
  - `nape - Enzima de Síntese de AEA`  
  - `faah - Enzima de Degradação de AEA`

---

## 📂 Estrutura do Projeto



├── Main.py # Script principal do projeto

├── melhores_hyperparametros_OX_50_MSE_.json # Hiperparâmetros salvos (necessário para execução)

├── melhor_modelo_OX_50_MSE.pth # Checkpoint do modelo treinado (necessário para execução)

├── requirements.txt # Dependências do projeto

└── README.md # Documentação


---

## ⚙️ Pré-requisitos

- Python 3.8+
- GPU CUDA (opcional, o código também roda em CPU)

Instale as dependências com:

```bash
pip install -r requirements.txt

▶️ Como Executar

Certifique-se de que os arquivos de pesos (.pth) e hiperparâmetros (.json) estejam no mesmo diretório do script.

Rode o script principal:

python main.py


O programa irá:

Gerar dados simulados de entrada

Exibir os dados simulados

Realizar a predição com o modelo carregado

Exibir as predições dos biomarcadores na escala original


📌 Observações

Caso os arquivos de pesos ou hiperparâmetros não estejam presentes, o script exibirá um erro informando quais arquivos estão faltando.

Você pode modificar a quantidade de amostras simuladas alterando a linha no final do script:

predizer_com_modelo_salvo(n_amostras=10)`

🛠️ Tecnologias Utilizadas

PyTorch
 - Framework de Deep Learning

Scikit-learn
 - Pré-processamento de dados

Pandas
 - Manipulação de dados

NumPy
 - Operações numéricas

👨‍💻 Autor

Projeto desenvolvido por Vinicius Mantovam
MBA em Data Science e Analytics - USP/Esalq

Você pode ouvir a apresentação do meu projeto [aqui](https://cdn.discordapp.com/attachments/623300452552802305/1411083495006015669/Projeto_mestrado_Vinicius_Mantovam_USP.mp3).


OBS: A rede não desempenhou poder preditivo satisfatório,apesar da realização de engenharia das features e tratamento dos dados. O que é esperado em um ecossiteama complexo como o corpo humano!