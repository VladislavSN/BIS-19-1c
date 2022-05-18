from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from IPython.display import Image
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

data = pd.read_csv('../content/drive/MyDrive/A_Z Handwritten Data.csv')

print(data.shape) # (372451, 785)

data.rename(columns={'0':'label'}, inplace=True)
print(data.head())

X = data.drop('label',axis = 1)
y = data['label']

(X_train, X_test, Y_train, Y_test) = train_test_split(X, y)

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

print(X_train.shape) # (1, 249542, 784, 1)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(Y_test.shape) # (122909, 26)

num_classes = Y_test.shape[1] # 26

model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', data_format="channels_last", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=256)

# Процент ошибок 
scores = model.evaluate(X_test,Y_test)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save_weights("model.h5")

#Тестирование изображения

img_path = '../content/R.png'
Image(img_path, width=28, height=28)

img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")

# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 28, 28, 1)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

prediction = model.predict(x)

prediction = np.argmax(prediction)
print("Номер класса:", prediction)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print("Название класса:", classes[prediction])

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title(classes[prediction])
plt.axis('off')