import csv
import matplotlib.image as mpimg
from collections import namedtuple
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_dir = 'data'

Sample = namedtuple('Sample', ['image', 'flip', 'measurement'])

def genAugmentedSamples(data_dir, line, augment=True, correction=0.1):
    measurement = float(line[3])
    sub_samples = []
    positions = ['center']
    flips = [False]
    if augment:
        positions.extend(['left', 'right'])
        flips.append(True)
    for flip in flips:
        for pos in positions:
            if pos == 'center':
                img_name = line[0]
                new_measurement = measurement
            elif pos == 'left':
                img_name = line[1]
                new_measurement = measurement + correction
            else:
                img_name = line[2]
                new_measurement = measurement - correction
            sub_samples.append(Sample(data_dir + '/' + img_name.lstrip(), flip, new_measurement))
    return sub_samples

def readSamples(data_dir, augment=True):
    samples = []
    with open(data_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        first_line_skip = True
        for line in reader:
            if first_line_skip:
                first_line_skip = False
            else:
                samples.extend(genAugmentedSamples(data_dir, line, augment))
    return samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for sample in batch_samples:
                image = mpimg.imread(sample.image)
                measurement = sample.measurement
                if sample.flip:
                    image = np.fliplr(image)
                    measurement = - measurement
                images.append(image)
                measurements.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def numberOfBatches(num_samples, batch_size=32):
    num_batches = num_samples / batch_size
    if (num_samples%batch_size) != 0:
        num_batches += 1
    return num_batches

samples = readSamples('data', augment=True)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
BATCH_SIZE = 128
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

steps_per_epoch = numberOfBatches(len(train_samples), BATCH_SIZE)
validation_steps = numberOfBatches(len(validation_samples), BATCH_SIZE)

#nvnet model
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation=None))
model.add(Activation('relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation=None))
model.add(Activation('relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation=None))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=None))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=None))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100, activation=None))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50, activation=None))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10, activation=None))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, epochs=5)

model.save('model.h5')
