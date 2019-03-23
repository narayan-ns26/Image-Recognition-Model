#building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the CNN
classifier = Sequential()

#convolution step
classifier.add(Convolution2D(32 ,3 ,3 ,input_shape = (64 ,64 ,3) ,activation = 'relu')) 
#32-No. of filters,3*3 matrix,color image-3(r,g,b),64*64pixel

#pooling step
classifier.add(MaxPooling2D(pool_size = (2 ,2)))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim = 128,activation='relu'))
classifier.add(Dense(output_dim = 1,activation='sigmoid'))
#from IPython.display import display
from PIL import Image
#compilingthecnn
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])

#fittingthecnntotheimages
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')
classifier.fit_generator(training_set,
               samples_per_epoch=8000,
               nb_epoch=25,
               validation_data=test_set,
               nb_val_samples=2000)