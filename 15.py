import numpy as np
dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')

X = dataset[:,1:785]
Y = dataset[:,0]

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, shuffle=True)


import matplotlib.pyplot as plt

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

for i in range(40):
    x = x_train[i]
    x = x.reshape((28, 28))
    plt.axis('off')
    im = plt.subplot(5, 8, i+1)
    plt.title(word_dict.get(y_train[i]))
    im.imshow(x, cmap='gray')
plt.show()


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

from keras.utils import to_categorical
train_labels = to_categorical(y_train, 26)
test_labels = to_categorical(y_test, 26)



from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(26, activation='softmax'))


# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(x_train, train_labels,
#                     validation_data=(x_test, test_labels),
#                     epochs=15, batch_size=256)
#

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(x_train, train_labels,
                    validation_data=(x_test, test_labels),
                    epochs=15, batch_size=256)


test_loss, test_acc = model.evaluate(x_test, test_labels)
print(f'Точность на тестовом образцу: {test_acc:.3%}')
print(f'Потери на тестовом образце: {test_loss:.3%}')

n = 100
x = x_test[n]

print(x.shape)

import numpy as np
x = np.expand_dims(x, axis=0)
print(x.shape)

prediction = model.predict(x)
print(f'Вектор результата на 26 нейронах: {prediction}')
import numpy as np
pred = np.argmax(prediction)
print(f'Распознана буква: {word_dict.get(pred)}')
print(f'Правильное значение: {word_dict.get(np.argmax(test_labels[n]))}')




history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'b-.', label= 'Потери на обучаюзей выборкуе')
plt.plot(epochs, val_loss_values, 'b', label = 'Потери на тестовой выборке')
plt.title('График потерь')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

history_dict = history.history
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'g-.', label='Точность на обучающей выборке')
plt.plot(epochs, val_acc_values, 'g', label='Точность на тестовой выборке')
plt.title('График точности')
plt.xlabel('Эпоха обучения')
plt.ylabel('Точность')
plt.legend()
plt.show()