# part-1 creating CNN

#importing all keras packages and libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize CNN
classifier = Sequential()

# step-1 convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step-3 flattening
classifier.add(Flatten())

# step-4 full connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# compilling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# part-2 fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                '/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=25,
        validation_data=test_set,
        validation_steps=20)

# import Image class
from PIL import Image
import numpy as np

img = Image.open('/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/cat.4003.jpg')
img = img.convert('RGB')
img = img.resize((64, 64), 0)
x = np.asarray(img)
x = x[np.newaxis, ...]
print(x.shape)

y_pred = classifier.predict(x)
print(y_pred)


# save model
from keras.models import load_model
from keras.models import h5py

classifier.save('/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/my_model.h5')  # creates a HDF5 file 'my_model.h5'
del classifier  # deletes the existing model

# returns a compiled model
# identical to the previous one
classifier = load_model('/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/my_model.h5')
y_pred = classifier.predict(x)

print(y_pred)