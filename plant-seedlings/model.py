import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scikeras.wrappers import KerasClassifier

import config

def cnn_model(
        kernel_size=(3, 3),
        pool_size=(2, 2),
        first_filters=32,
        second_filters=64,
        third_filters=128,
        dense_units=256,
        dropout_conv=0.3,
        dropout_dense=0.3,
        image_size=50
):
    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                     input_shape = (image_size, image_size, 3)))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    #model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size)) 
    model.add(Dropout(dropout_conv))
    
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    #model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))
    
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    #model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))
    
    model.add(Flatten())
    model.add(Dense(dense_units, activation = "relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(12, activation = "softmax"))

    model.compile(
        Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

checkpoint = ModelCheckpoint(
    config.MODEL_PATH,
    monitor='accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
reduce_lr = ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.5,
    patience=2,
    verbose=1,
    mode='max',
    min_lr=1e-5
)
callbacks_list = [checkpoint, reduce_lr]

cnn_clf = KerasClassifier(
    build_fn=cnn_model,
    batch_size=config.BATCH_SIZE,
    validation_split=0.1,
    epochs=config.EPOCHS,
    verbose=2,
    callbacks=callbacks_list,
    image_size=config.IMAGE_SIZE
)

if __name__=='__main__':
    model = cnn_model()
    model.summary()
    