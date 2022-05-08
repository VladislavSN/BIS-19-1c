import os 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pathlib
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_animal_crossing = 'animal_crossing'            #Директория для animal_Crossing
file_doom = 'doom'                                  #Директория для animal_Crossing
data_dir = pathlib.Path('C://123//TestII//test')    #Путь с изображениями для теста
batch_size=50   
image_shape=150

classes = ['animal_crossing', 'doom']               

model = load_model('doomanimal_7.h5')               #Загрузка модели 

test_image_generator = ImageDataGenerator(rescale=1./255)
#Загрузка обработанных изображений 
test_dataset = test_image_generator.flow_from_directory('C://123//TestII//test_images', batch_size=batch_size, target_size=(image_shape, image_shape))
images = list(data_dir.glob('*.*'))

prediction = model.predict(test_dataset)
#Перемещение изображений по директориям
for i in range(len(prediction)):
    prediction_img = np.argmax(prediction[i])
    print(images[i].name, ":")
    print("Номер класса: ", prediction_img)
    print("Название класса: ", classes[prediction_img])
    if (prediction_img == 0 ):
        shutil.copy(os.path.join(data_dir, images[i]), file_animal_crossing)
    else:  shutil.copy(os.path.join(data_dir, images[i]), file_doom) 

