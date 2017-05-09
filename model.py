import csv
import matplotlib.image as mpimg
from collections import namedtuple
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

Sample = namedtuple('Sample', ['image', 'flip', 'measurement'])

def generateSamples(data_dir, line, augment=True, correction=0.1):
    """
    It generates a list of samples with existing samples after optional
    augmentation. Each sample is a named tuple holding image location,
    'flip' state to flip the image after reading in and steering angle
    measurement that corresponds to this sample. 

    Args:
        data_dir: string, name of local directory for images.
        line: list, individual line read from driving_log.csv file
        augment: bool, whether or not left and right camera images
        will be added to samples along with flipping the images.
        correction: float, steering angle correction for left and right
        images
    """

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
            if flip:
                new_measurement = - new_measurement
            sub_samples.append(Sample(data_dir + '/' + img_name.lstrip(), flip, new_measurement))
    return sub_samples

def readSamples(data_dir, augment=True):
    """
    It reads each line from driving_log.csv and calls generateSamples
    for each data piece to create a list of samples to return.
    Args:
        data_dir: string, local directory for data
        augment: bool, to augment samples with left and right camera images
        with flipping.
    """
    samples = []
    with open(data_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        first_line_skip = True
        for line in reader:
            if first_line_skip:
                first_line_skip = False
            else:
                samples.extend(generateSamples(data_dir, line, augment))
    return samples

def dataGenerator(samples, batch_size=32):
    """
    It is a python generator for samples to feed a batch-size of data
    samples at a time to keras' training phase.
    Args:
        samples: list of tuples to feed the generator
        batch_size: int, size of each batch training works on at a time
    """
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
                images.append(image)
                measurements.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def numberOfBatches(num_samples, batch_size=32):
    """
    It computes number of batches given total number of samples and batch size.
    Args:
        num_samples: int, total number of samples
        batch_size: int, batch size
    """
    num_batches = num_samples / batch_size
    if (num_samples%batch_size) != 0:
        num_batches += 1
    return num_batches

def nvDriveNetModel():
    """
    It creates a neural network in Keras based on Nvidia's drivenet model.
    It first normalizes the input image and crops top and bottom pixels.
    Then, it uses drivement layers with relu as activation. Finally, it makes
    use of dropout in the penultimate layer to reduce overfitting.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation=None))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return  model

# Keep training and validation samples ready
samples = readSamples('data', augment=True)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Setup generators for training and validation samples
BATCH_SIZE = 64
train_generator = dataGenerator(train_samples, batch_size=BATCH_SIZE)
validation_generator = dataGenerator(validation_samples, batch_size=BATCH_SIZE)

steps_per_epoch = numberOfBatches(len(train_samples), BATCH_SIZE)
validation_steps = numberOfBatches(len(validation_samples), BATCH_SIZE)

# Create model and train it using above generators
model = nvDriveNetModel()
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, epochs=5)

# Save the model
model.save('model.h5')
