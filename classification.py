# Importação das bibliotecas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil

if __name__ == '__main__':

# Pré-processamento
    ## Define transformações para pré-processamento das imagens do dataset para o conjunto de treinamento e teste
        ###Redimensiona a imagem para o tamanho de entrada do modelo InceptionV3
        ###No conjunto de treinamento executa um flip horizontal aleatório na imagem com uma probabilidade de 0,5 para aumentar o dataset e evitar vieses 
        ###Converte a imagem para um Tensor PyTorch
        ###Normaliza os valores dos canais da imagem
        ###No conjunto de teste executa um corte de 299x299 pixel para garantir que a área de interesse esteja presente no teste

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


        ### Diretório raiz do dataset
    data_root = 'images'

        ### Diretório onde será criado as pastas para treinamento e teste
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

        ### Monta uma lista das classes/frutas
    classes = os.listdir(data_root)

        ### Loop que percorre cada classe/fruta
    for class_name in classes:
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        ### Criar as pastas para treinamento e teste de cada classe
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        ### Lista as imagens da classe
        images = os.listdir(class_dir)

        ### Embaralha as imagens
        random.shuffle(images)

        ### Divide as imagens do dataset em 80% para treinamento e 20% para teste
        num_images = len(images)
        num_train = int(0.8 * num_images)
        num_test = num_images - num_train

        ### Loops que copiam as imagens para as pastas de treinamento e teste
        train_images = images[:num_train]
        test_images = images[num_train:]

        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_class_dir, image)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_class_dir, image)
            shutil.copy(src, dst)

# Preparações para o treinamento
    ## Hiperparâmetros do treinamento
    batch_size = 32
    epocas = 10
    num_classes = 9

    ## Cria DataLoader para carregar os dados de treinamento e teste
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test']),
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4),
    }

    ## Utilizar a GPU se estiver disponível
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Carregar o modelo pré-treinado InceptionV3
    model = models.inception_v3(weights="Inception_V3_Weights.IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    ## Função de perda e otimizador
    criterio = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    ## Nome das classes
    class_names = image_datasets['train'].classes
    
 # Treinamento   
    ## Função de treinamento do modelo
    def train_model(model, criterio, optimizer, scheduler, epocas):
        model.train()
        for epoch in range(epocas):
            print(f'Epoch {epoch+1}/{epocas}')
            print('-' * 10)
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    logits = outputs.logits
                    _, preds = torch.max(logits, 1)
                    loss = criterio(logits, labels)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                ### Obter as classes das imagens atuais
                image_classes = [class_names[label] for label in labels]

                ### Imprimir as classes e as imagens
                for i in range(len(inputs)):
                    print(f'Class: {image_classes[i]}, Image: {image_datasets["train"].imgs[i][0]}')

            epoch_loss = running_loss / len(image_datasets['train'])
            epoch_acc = running_corrects.double() / len(image_datasets['train'])

            print(f'Perda do Treinamento: {epoch_loss:.4f} Acurácia do Treinamento: {epoch_acc:.4f}')

            scheduler.step()

    ## Chamada da função de treinamento
    train_model(model, criterio, optimizer, scheduler, epocas)

# Teste
    ## Função para calcular e exibir a matriz de confusão
    def plot_confusion_matrix(cm, classes):
        plt.figure(figsize=(7, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Classe Prevista')
        plt.ylabel('Classe Real')

        # Adicionando os valores dentro da matriz
        thresh = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.show()

    # Avaliar o modelo no conjunto de teste
    model.eval()
    test_corrects = 0
    test_predictions = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        test_corrects += torch.sum(preds == labels.data)
        test_predictions.extend(preds.tolist())
        
        
    test_accuracy = test_corrects.double() / len(image_datasets['test'])
    print(f'Acurácia do Teste: {test_accuracy:.4f}')

    # Calcular a matriz de confusão
    test_labels = image_datasets['test'].targets
    confusion = confusion_matrix(test_labels, test_predictions)

    # Exibir a matriz de confusão
    plot_confusion_matrix(confusion, class_names)