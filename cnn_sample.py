#!/usr/bin/python
import numpy as np
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected1D, BatchNormalization
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.utils import np_utils

def train(train):
    train = np.array(train)
    train = train.astype('float32')
    train = train.reshape(train.shape[0], train.shape[1], 1)

    y = pd.read_csv('y_train_path')['Class'].values
    y = y-1#vector
    y = np_utils.to_categorical(y, 9)#matrix


    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    ckpt = ModelCheckpoint('best_model_saving_path', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    tb= TensorBoard(log_dir = 'log_saving_dir' + model_stat, histogram_freq=0, write_graph=True, write_images=True)

    model.fit(train, y, batch_size=10, epochs=1000, validation_split = 0.2, callbacks = [tb, ckpt])
    model.save('final_model_saving_path')

def predict(test, model,):
    test = np.array(test)
    test = test.astype('float32')
    test = test.reshape(test.shape[0], test.shape[1], 1)

    pid = pd.read_csv('variants_path')['ID'].values

    pred = model.predict(test)

    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('output', index=False)



