
# Classificação de Frutas com a Rede Neural Convolucional InceptionV3

Uma breve descrição sobre o que esse projeto faz e para quem ele é


## Equipe
Nathan Patrike da Luz Weber

## Descrição do projeto

Este projeto tem como objetivo classificar um conjunto de imagens de frutas utilizando o modelo de rede neural convolucional InvecptionV3

O algoritmo foi implementado com a linguagem de programação Python e o framework Pytorch
## Dataset
Fruits Dataset (Images)

Disponível em: <https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images?resource=download>

O conjunto de dados contém 360 imagens de 9 tipos diferentes de frutas: maçã, banana, cereja, chickoo, uva, kiwi, manga, laranja e morango. Cada tipo de fruta tem 40 imagens. As imagens estão no formato PNG e possuem diferentes dimensões.
## Repositório do projeto
GitHub: <https://github.com/NathanWeber/fruit-classification>
## Classificador e acurácia

Para classificação das imagens foi usado o modelo de rede neural convolucional InceptionV3 com o framework PyTorch

As imagens do dataset foram divididas em 80% para treinamento e 20% para teste

As imagens também passaram por um pré-processamento:

- Redimensionado as imagens para o tamanho de entrada do modelo InceptionV3 (299x299 pixel)
- No conjunto de treinamento executa um flip horizontal aleatório na imagem com uma probabilidade de 0,5 para aumentar o dataset e evitar vieses 
- Convertido as imagens para um Tensor PyTorch
- Normalizado os valores dos canais das imagens 
- No conjunto de teste executado um corte de 299x299 pixel para garantir que a área de interesse esteja presente no teste

A rede foi inicializada de forma pré-treinada com os pesos do treinamento executado por ela no dataset ImageNet

Não foi alterada nenhuma camada da rede, exceto a camada totalmente conectada para uma camada linear com o número de 9 classes do problema em questão

Para o treinamento foram utilizados os seguintes hiperparâmetros:
    
    #Tamanho do lote
    batch_size = 32

    #Épocas de treinamento
    epocas = 10

    #Taxa de aprendizado
    lr=0.001

    #Momentum
    momentum=0.9

A função de perda utilizada é a função CrossEntropyLoss

A função de otimização utilizada é a função Stochastic Gradient Descent

Com essas definções, ao fim da execução o algoritmo obteve as seguintes acurácias:

Acurácia do Treinamento: 0.9164
Acurácia do Teste: 0.9617
## Instalação e Execução

As etapas abaixo são necessárias antes da execução do algoritmo:

1) Efetuar o download do projeto disponível na Seção Repositório e descompactar o arquivo se necesário
2) Efetuar o download do conjunto de imagens disponível na Seção Dataset e descompactar o arquivo. Realizar a copia por completo da pasta **images** e colar no diretório raiz do projeto
3) Possuir o Python 3.11.4 ou superior
4) Possuir o Pip (Gerenciador de pacotes) 23.1.2 ou superior
5) Installar as bibliotecas descritas no arquivo **requirements.txt** . É possível na pasta raiz do projeto executar o comando **pip install -r requirements.txt** ou instalar cada uma das bibliotecas manualmente com o comando **pip install *nome_biblioteca***


O projeto está dividido na seguinte estrutura:

- README.md : arquivo com as informações sobre o projeto
- video.txt : contém o link para um video explicativo do projeto
- requirements.txt  : bibliotecas necessárias
- classification.py: algoritmo que classifica as imagens do dataset

O dataset está dividido na seguinte estrutura:

- images: Arquivo original com as imagens das frutas
- dataset (Criado na execução do algoritmo): Uma pasta *train* para o conjunto de treinamento e uma pasta *test* para o conjunto de teste, as quais possuem outras pastas cada uma sendo uma classe que possui as imagens das frutas


Após isso é possível executar o arquivo **classification.py** para iniciar o algoritmo

## Instruções de uso

Através do link: <https://drive.google.com/file/d/1uQ7dyUt1NlTqtovnTdJAfSJEWk-V3mhF/view?usp=sharing> é possível acessar um video explicativo sobre o projeto

Antes da execução do algoritmo é possível alterar alguns hiperparâmetros do treinamento do algoritmo, através das seguintes variáveis

    #Tamanho do lote
    batch_size = 32

    #Épocas de treinamento
    epocas = 10

    #Taxa de aprendizado
    lr=0.001

    #Momentum
    momentum=0.9


Com a execução do algoritmo será iniciado a criação do dataset e divisão das imagens em conjuntos de treinamento e teste. Caso ocorrer algum erro, revisar os diretórios nas seguintes linhas do código:

    ### Diretório raiz do dataset
    data_root = 'images'

    ### Diretório onde será criado as pastas para treinamento e teste
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'


Ao fim da execução é impresso no terminal a acurácia do treinamento e do teste

É plotado também uma imagem da matriz de confusão