import tensorflow as tf
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random

def generator(csv_data, batch_size):

  while True:

    num_data = len(csv_data)
    positions=['left','center','right']
    corrections=[0.25, 0 ,-0.25]
    #csv_data.sample(frac=1)
    # itrate through batches
    for start in range(0, num_data, batch_size):
      images, measurements = [], []
      # itrate inside the batch
      for csv_line in range(start, start + batch_size):
        if csv_line < num_data:
          index = csv_data.index[csv_line]
          r=random.choice([0,1,2])
          measurement = csv_data['steering'][index]+corrections[r]
          path = PATH + csv_data[positions[r]][index][1:]
          if r == 1: path = PATH + csv_data[positions[r]][index]
          image = imread(path)
          if random.random() > 0.5:
            measurements.append(-measurement)
            images.append(np.fliplr(image))

      yield np.array(images), np.array(measurements)
    
model = Sequential()

model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample = (2, 2), border_mode = "same"))
model.add(Flatten())
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

#model.summary()
model.compile(optimizer = "adam", loss = "mse")

batch_size = 28
epochs = 2
PATH = "data/"
csv_name = "driving_log.csv"

X_data = pd.read_csv(PATH + csv_name, usecols = [0, 1, 2, 3])

X_train, X_valid = train_test_split(X_data, test_size = 0.20)
num_train = len(X_train)
num_valid = len(X_valid)
#X_data.head()

print('Training model...')

training_generator = generator(X_train, batch_size)
validation_generator = generator(X_valid, batch_size)



history_object = model.fit_generator(training_generator,
                 samples_per_epoch = num_train,
                 validation_data = validation_generator,
                 nb_val_samples = num_valid,
                 nb_epoch = epochs,
                 verbose = 1)

print('Saving model...')
model.save("model.h5")
#with open("model.json", "w") as json_file:
  #json_file.write(model.to_json())
print("Model Saved.")