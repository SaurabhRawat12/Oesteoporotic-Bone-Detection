# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:13:48 2019

@author: DELL
"""

# -*- coding: utf-8 -*-
"""mura_trail_01.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GYSEcBR3QLrvFo3tL4Q8P1qbLrttRiEm
"""

from google.colab import drive
drive.mount('/content/drive')

from keras.layers import Input, Lambda, Dense, Flatten,MaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,Adam
from keras.metrics import binary_crossentropy
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import numpy as np
tf.compat.v1.get_default_graph
from keras.layers.core import Dense, Activation,Dropout
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D,BatchNormalization

img_rows, img_cols = 224, 224

MobileNet = MobileNet(weights = "imagenet",include_top = False,
                      input_shape = (img_rows, img_cols, 3))

for layer in MobileNet.layers:
    layer.trainable = False
    
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
    
def addTopModelMobileNet(bottom_model, num_classes):
    '''Creates the top or head of the model that will be 
    placed ontop of the bottom layers'''
    
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation = "relu")(top_model)
    top_model = Dense(1024,activation = "relu")(top_model)
    top_model = Dense(512,activation = "relu")(top_model)
    top_model = Dense(2,activation = "softmax")(top_model)
    return top_model
    
num_classes = 2

FC_Head = addTopModelMobileNet(MobileNet,num_classes)
model = Model(inputs = MobileNet.input, outputs = FC_Head)
print(model.summary())

'''mobile = VGG16()
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

base_model=VGG16(weights='imagenet',include_top=False,pooling='max') #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=BatchNormalization(momentum=0.99, epsilon=0.001,scale=True)(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=BatchNormalization(momentum=0.99, epsilon=0.001,scale=True)(x)
x=Dense(512,activation='relu')(x)
x=Dense(512,activation='relu')(x) #dense layer
x=BatchNormalization(momentum=0.99, epsilon=0.001,scale=True)(x) 
x=Dense(256,activation='relu')(x)
x=Dense(256,activation='relu')(x) #dense layer
x=BatchNormalization( momentum=0.99, epsilon=0.001,scale=True)(x) 
x=Dense(128,activation='relu')(x)
x=Dense(128,activation='relu')(x) #dense layer
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation outupt is six
model=Model(inputs=base_model.input,outputs=preds)  #specify the inputs and output

model.summary()

for layer in model.layers[10:]:
    layer.trainable=True
for layer in model.layers[:10]:
    layer.trainable=False'''

'''train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip = True,vertical_flip=True,rotation_range=30) #included in our dependencies

train_generator=train_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/dataset/train',
                                                 target_size=(224,224),
                                                 #color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,)
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            #color_mode='grayscale',
                                            class_mode = 'categorical',shuffle = True,)'''

train_data_dir = "2nd evalution/mod_train"
validation_data_dir = "2nd evalution/mod_test"

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip = True,
        fill_mode = "nearest"
        )

validation_datagen = ImageDataGenerator(rescale = 1./255)

#  set our batch size(typically on most mid tier systems we will use 16-32)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = "categorical"
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode= "categorical"
        )

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
'''
his = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                    validation_data = test_set,
                   epochs=20)
'''

his = model.fit_generator(
        train_generator,
        steps_per_epoch = 135,
        epochs = 20,
        validation_data = validation_generator,
        validation_steps = 31
        )
plt.plot(his.history['loss'], label='train loss')
plt.plot(his.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(his.history['acc'], label='train acc')
plt.plot(his.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model_save_name = 'Mura_try_2.model'
path = F"/content/drive/My Drive/Colab Notebooks/final_UTKface/{model_save_name}"
model.save(path,overwrite=True)