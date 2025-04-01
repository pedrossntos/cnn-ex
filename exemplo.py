from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 
from keras.optimizers import Adam 
from keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.datasets import mnist  
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()  # carrega o conjunto de dados MNIST de treino e teste
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255  # reshape as imagens de treino e normaliza para intervalo [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255  # mesma coisa para as imagens de teste
y_train = to_categorical(y_train, 10)  # converte as labels de treino para one-hot encoding (10 classes)
y_test = to_categorical(y_test, 10)  # converte as labels de teste para one-hot encoding

model = Sequential()  # cria um modelo sequencial (camadas empilhadas)
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))  # primeira camada convolucional, 32 filtros de 5x5, ativação ReLU
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))  # segunda camada convolucional com 64 filtros
model.add(MaxPooling2D(pool_size=(2, 2)))  # camada de pooling para reduzir a resolução das imagens
model.add(Dropout(0.25))  # aplica dropout de 25% para evitar overfitting
model.add(Flatten())  # achata a saída da camada anterior para uma dimensão unidimensional
model.add(Dense(128, activation='relu'))  # camada densa com 128 neurônios e ativação ReLU
model.add(Dropout(0.5))  # aplica dropout de 50% após a camada densa
model.add(Dense(10, activation='softmax'))  # camada de saída com 10 neurônios, uma para cada classe, e ativação softmax para classificação multi-classe

optimizer = Adam() 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # compila o modelo com a função de perda e métrica de acurácia
print(model.summary())  # imprime o resumo do modelo

plot_model(model, to_file='model_architecture_graphviz.png', show_shapes=True, show_layer_names=True)


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',  # cria o callback para reduzir a taxa de aprendizado
                                            factor=0.5,  # reduz a taxa de aprendizado pela metade
                                            verbose=1,  # imprime uma mensagem quando a taxa for reduzida
                                            patience=3,  # espera 3 épocas para reduzir a taxa de aprendizado
                                            min_lr=0.00001)  # limita a taxa de aprendizado mínima para 0.00001
batch_size = 32  # define o tamanho do batch para o treinamento
epochs = 10  # define o número de épocas

history = model.fit(x_train, y_train,  # treina o modelo
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=0.2,  # usa 20% dos dados de treino para validação
                    callbacks=[learning_rate_reduction],  # inclui o callback para redução da taxa de aprendizado
                    verbose=1)  # imprime o progresso do treinamento

history_dict = history.history  # obtém o histórico de treinamento
acc = history_dict['accuracy']  # extrai a acurácia do treinamento
val_acc = history_dict['val_accuracy']  # extrai a acurácia da validação
range_epochs = range(1, len(acc) + 1)  # cria uma sequência para o número de épocas

plt.style.use('default')
accuracy_val = plt.plot(range_epochs, val_acc, 'bo', label='Acurácia no conjunto de validação')
accuracy_train = plt.plot(range_epochs, acc, 'b', label='Acurácia no conjunto de treinamento')
plt.setp(accuracy_val, markersize=5, linewidth=2)
plt.setp(accuracy_train, markersize=5, linewidth=2) 
plt.xlabel('Épocas') 
plt.ylabel('Acurácia')  
plt.legend(loc='lower right') 
plt.title('Acurácia do modelo')  
plt.show() 
