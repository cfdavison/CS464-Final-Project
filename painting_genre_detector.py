# A python version of the Casey's jupytr notebook 
# adapted to test the parameters of the model.
# Written by Creston Davison

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import os
import time
import numpy as np
import matplotlib.pyplot as plt

labels = ['Abstractism', 'Baroque', 'Byzantine', 'Cubism', 'Expressionism', 'High_Renaissance',
             'Impressionism', 'Mannerism', 'Muralism', 'Northern_Renaissance', 'Pop_Art',
             'Post-Impressionism', 'Primitivism', 'Proto_Renaissance', 'Realism', 'Romaticism',
             'Suprematism', 'Surrealism', 'Symbolism']

# Generate models with a given image dimension and category count
# Returns a list of models
def generate_models(image_dim, categories=19):
    model_1_desc = 'Conv2D->MaxPool2d->Conv2D->MaxPool2d->Conv2d'
    model_1 = models.Sequential()
    model_1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_dim, image_dim, 3)))
    model_1.add(layers.MaxPooling2D((2, 2)))
    model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_1.add(layers.MaxPooling2D((2, 2)))
    model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_1.add(layers.Dropout(0.2))
    model_1.add(layers.Flatten())
    model_1.add(layers.Dense(categories, activation='softmax'))
    print('Model Generated')

    return [(model_1, model_1_desc)]

# Callback class for the models to print out how long each epoch takes
class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Create the generators to read the image files for the training and validation data
# returns a touple (training_data_generator, validation_data_generator)
def get_generators(image_dim, batch_size, data_directory):
    data_generator = ImageDataGenerator(
        validation_split=0.25,
        rescale=1.0/255.0,
        horizontal_flip=False,
        vertical_flip=False)

    # ImageDataGenerator returns an interable for the image files
    train_data = data_generator.flow_from_directory(
        directory=data_directory,
        class_mode='categorical',
        target_size=(image_dim, image_dim),
        batch_size=batch_size,
        subset="training",
        shuffle=True,
        classes=labels)

    validation_data = data_generator.flow_from_directory(
        directory=data_directory,
        class_mode='categorical',
        target_size=(image_dim, image_dim),
        batch_size=batch_size,
        subset="validation",
        shuffle=True,
        classes=labels)

    return (train_data, validation_data)

# Model Testing
def test_models():
    # PARAMS
    test_cases = [ 
        {"image_dim": 128, "epoch_count": [4,5,6], "batch_size": [4,8,16]},
        {"image_dim": 224, "epoch_count": [4,5,6], "batch_size": [4,8,16]},
        {"image_dim": 256, "epoch_count": [4,5,6], "batch_size": [4,8,16]}
    ]
        
    data_directory = './data/genres'
    model_directory = './models'
    results_directory = './results'
    index = 0
    while os.path.exists(results_directory + "/run_%s" % index):
        index += 1

    for case in test_cases:
        for b in case['batch_size']:
            for e in case['epoch_count']:
                i = case['image_dim']
                models = generate_models(i)
                for m in models:
                    print('model_%s_%s_%s' % (i, b, e))
                    train_data, validation_data = get_generators(i, b, data_directory)
                    STEP_PER_EPOCH_TRAIN = train_data.n//train_data.batch_size
                    STEP_PER_EPOCH_VALID = validation_data.n//validation_data.batch_size
                    time_callback = TimeHistory()
                    
                    m[0].compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

                    m[0].fit(
                        x=train_data,
                        epochs=e,
                        batch_size=b,
                        callbacks=[time_callback],
                        steps_per_epoch=STEP_PER_EPOCH_TRAIN
                    )

                    times = time_callback.times
                    train_acc = m[0].evaluate(train_data,batch_size=b, steps=STEP_PER_EPOCH_TRAIN, return_dict=True)['accuracy'] * 100.0
                    test_acc = m[0].evaluate(validation_data,batch_size=b, steps=STEP_PER_EPOCH_VALID, return_dict=True)['accuracy'] * 100.0

                    res = 'Model %s\n\tImage Dimensions: %d\n\tEpochs: %d\n\tBatch Size: %d\n\t' % (m[1], i, e, b)
                    res += 'Training Accuracy: %0.2f%%\n\t' % train_acc
                    res += 'Test Accuracy: %0.2f%%\n\t' % test_acc
                    res += 'Epoch Times: %s\n' % (' '.join(str(t) for t in times))

                    res += '\n'

                    file = open(results_directory + "/run_%s" % index, 'a')
                    file.write(res)
                    file.close()

                    print(res)
                    m[0].save(model_directory + "/model_%s_%s_%s" % (i, b, e))
                    print('Model saved to ' + model_directory + "/model_%s_%s_%s" % (i, b, e))
    print("Test results saved to /run_%s" % index)

test_models()
print("Done!")


            
        

        









