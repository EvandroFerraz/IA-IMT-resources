### Criamos várias funções úteis ao longo do curso.
### Armazenando-as aqui para ficarem facilmente acessíveis.

import tensorflow as tf

# Cria uma função para importar uma imagem e redimensioná-la para uso no nosso modelo
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Lê uma imagem a partir de filename, transforma em tensor e remodela para
  (224, 224, 3).

  Parâmetros
  ----------
  filename (str): nome do arquivo da imagem-alvo
  img_shape (int): tamanho para redimensionar a imagem-alvo, padrão 224
  scale (bool): se deve escalar os pixels para o intervalo (0, 1), padrão True
  """
  # Ler a imagem do disco
  img = tf.io.read_file(filename)
  # Decodificar para um tensor
  img = tf.image.decode_jpeg(img)
  # Redimensionar a imagem
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Reescalar a imagem (obter todos os valores entre 0 e 1)
    return img/255.
  else:
    return img

# Observação: O código abaixo da matriz de confusão é uma adaptação da função
# plot_confusion_matrix do Scikit-Learn - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Nossa função precisa de um nome diferente de plot_confusion_matrix do sklearn
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Gera uma matriz de confusão rotulada comparando previsões e rótulos verdadeiros.

  Se classes for fornecido, a matriz de confusão será rotulada; caso contrário,
  valores inteiros de classe serão usados.

  Args:
    y_true: Array de rótulos verdadeiros (mesmo formato de y_pred).
    y_pred: Array de rótulos previstos (mesmo formato de y_true).
    classes: Array de rótulos de classe (ex.: em formato string). Se `None`, rótulos inteiros são usados.
    figsize: Tamanho da figura de saída (padrão=(10, 10)).
    text_size: Tamanho do texto na figura de saída (padrão=15).
    norm: normalizar valores ou não (padrão=False).
    savefig: salvar a matriz de confusão em arquivo (padrão=False).
  
  Returns:
    Um gráfico de matriz de confusão rotulada comparando y_true e y_pred.

  Exemplo de uso:
    make_confusion_matrix(y_true=test_labels, # rótulos verdadeiros do teste
                          y_pred=y_preds, # rótulos previstos
                          classes=class_names, # array com nomes das classes
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Criar a matriz de confusão
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalizar
  n_classes = cm.shape[0] # descobrir o número de classes com que estamos lidando

  # Plotar a figura e deixá-la apresentável
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # as cores representam quão 'correta' está a classe; mais escuro == melhor
  fig.colorbar(cax)

  # Existe uma lista de classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Rotular os eixos
  ax.set(title="Matriz de Confusão",
         xlabel="Rótulo previsto",
         ylabel="Rótulo verdadeiro",
         xticks=np.arange(n_classes), # criar espaço suficiente no eixo para cada classe
         yticks=np.arange(n_classes), 
         xticklabels=labels, # eixos rotulados com nomes das classes (se existirem) ou inteiros
         yticklabels=labels)
  
  # Fazer os rótulos do eixo x aparecerem na parte inferior
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Definir o limiar para as diferentes cores
  threshold = (cm.max() + cm.min()) / 2.

  # Escrever o texto em cada célula
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              alinhamento_horizontal := "center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Salvar a figura no diretório de trabalho atual
  if savefig:
    fig.savefig("confusion_matrix.png")
  
# Criar uma função para prever em imagens e plotá-las (funciona com múltiplas classes)
def pred_and_plot(model, filename, class_names):
  """
  Importa uma imagem localizada em filename, faz uma previsão com
  um modelo treinado e plota a imagem com a classe prevista como título.
  """
  # Importar a imagem-alvo e pré-processá-la
  img = load_and_prep_image(filename)

  # Fazer a previsão
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Obter a classe prevista
  if len(pred[0]) > 1: # checar se é multiclasse
    pred_class = class_names[pred.argmax()] # se houver mais de uma saída, pegar a de maior valor
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # se houver apenas uma saída, arredondar

  # Plotar a imagem e a classe prevista
  plt.imshow(img)
  plt.title(f"Previsão: {pred_class}")
  plt.axis(False);
  
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Cria uma instância de callback do TensorBoard para armazenar arquivos de log.

  Armazena os arquivos de log no caminho:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: diretório de destino para armazenar os logs do TensorBoard
    experiment_name: nome do diretório do experimento (ex.: efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Salvando arquivos de log do TensorBoard em: {log_dir}")
  return tensorboard_callback

# Plotar separadamente os dados de validação e de treinamento
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Retorna curvas separadas de loss para métricas de treinamento e validação.

  Args:
    history: Objeto History de um modelo TensorFlow (ver: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot de loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Épocas')
  plt.legend()

  # Plot de acurácia
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Acurácia')
  plt.xlabel('Épocas')
  plt.legend();

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compara dois objetos History de modelos TensorFlow.
    
    Args:
      original_history: Objeto History do modelo original (antes de new_history)
      new_history: Objeto History do treinamento contínuo (após original_history)
      initial_epochs: Número de épocas em original_history (o gráfico de new_history começa daqui)
    """
    
    # Obter as medidas do histórico original
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combinar o histórico original com o novo
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Gerar os gráficos
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Acurácia de Treinamento')
    plt.plot(total_val_acc, label='Acurácia de Validação')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Início do Fine Tuning') # recentrar o gráfico em torno das épocas
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treinamento e Validação')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Loss de Treinamento')
    plt.plot(total_val_loss, label='Loss de Validação')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Início do Fine Tuning') # recentrar o gráfico em torno das épocas
    plt.legend(loc='upper right')
    plt.title('Loss de Treinamento e Validação')
    plt.xlabel('época')
    plt.show()
  
# Criar função para descompactar um arquivo .zip no diretório de trabalho atual
# (já que vamos baixar e descompactar alguns arquivos)
import zipfile

def unzip_data(filename):
  """
  Descompacta filename no diretório de trabalho atual.

  Args:
    filename (str): caminho para a pasta zip alvo a ser descompactada.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

# Percorrer um diretório de classificação de imagens e descobrir quantos arquivos (imagens)
# existem em cada subdiretório.
import os

def walk_through_dir(dir_path):
  """
  Percorre dir_path retornando seu conteúdo.

  Args:
    dir_path (str): diretório alvo
  
  Returns:
    Uma impressão contendo:
      número de subdiretórios em dir_path
      número de imagens (arquivos) em cada subdiretório
      nome de cada subdiretório
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"Há {len(dirnames)} diretórios e {len(filenames)} imagens em '{dirpath}'.")
    
# Função para avaliar: acurácia, precisão, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calcula acurácia, precisão, recall e f1-score de um modelo de classificação binária.

  Args:
      y_true: rótulos verdadeiros na forma de um array 1D
      y_pred: rótulos previstos na forma de um array 1D

  Retorna um dicionário com accuracy, precision, recall e f1-score.
  """
  # Calcular a acurácia do modelo
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calcular precisão, recall e f1-score usando "média ponderada"
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results