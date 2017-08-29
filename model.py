import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend

base_path = '.\\drive_data\\'
driving_log = pd.read_csv(base_path + '.\\driving_log.csv', header=None, 
              names=['center_image', 'left_image', 'right_image', 
              'steering_angle', 'throttle', 'break', 'speed'])

train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)

'''
Read the training images as RGB
'''
def read_img(img):
  image = cv2.imread(img)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

'''
All the data is read in a batch of 32, avoiding memory hog
'''
def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: #loop forever so the generator never terminates
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      fwd_imgs = batch_samples['center_image'].map(lambda x: base_path + '\\IMG\\' + x.split('\\')[-1])
      X_train = np.stack(fwd_imgs.apply(read_img))
      y_train = np.stack(batch_samples['steering_angle'])
      yield sklearn.utils.shuffle(X_train, y_train)
      
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

backend.set_image_dim_ordering('tf')

# The nVidia CNN Architecture
model = Sequential()
# Crop
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
# Normalize
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# Convolution
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
# Flatten
model.add(Flatten())
# Dense
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32, verbose=1, epochs=8)

model.save('model.h5')