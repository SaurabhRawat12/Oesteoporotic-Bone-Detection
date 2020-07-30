from keras.applications import MobileNet

# MobileNet was designed to work on 224 x 224 pixel input image size
img_rows, img_cols = 224, 224

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = "imagenet",
                      include_top = False,
                      input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# layers are set to trainable as True by default 
for layer in MobileNet.layers:
    layer.trainable = False
    
# Lets print our layers
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
    top_model = Dense(num_classes,activation = "softmax")(top_model)
    return top_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 2
num_classes = 2

FC_Head = addTopModelMobileNet(MobileNet,num_classes)
model = Model(inputs = MobileNet.input, outputs = FC_Head)
print(model.summary())

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "2nd evalution/train"
validation_data_dir = "2nd evalution/test"

# Data Augmentation 

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
        class_mode = "binary"
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode= "binary"
        )


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(16, 3, 3, input_shape = (32,32, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = RMSprop(lr=0.001) , loss = 'binary_crossentropy', metrics = ['accuracy'])

# Enter the number of  training and validation samples here
nb_train_samples = 98
nb_validation_samples = 10

# we only train 10 epochs
epochs = 10
batch_size = 16

history = classifier.fit_generator(
        train_generator,
        steps_per_epoch = 
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size
        )
































































