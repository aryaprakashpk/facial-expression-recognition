import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
%matplotlib inline
import os
import gc
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns




from google.colab import drive
drive.mount('/content/drive')


data_fer = pd.read_csv('/content/drive/MyDrive/DATASET/fer2013.csv')
data_fer.head()

idx_to_emotion_fer = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

X_fer_train, y_fer_train = np.rollaxis(data_fer[data_fer.Usage == "Training"][["pixels", "emotion"]].values, -1)
X_fer_train = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_train]).reshape((-1, 48, 48))
y_fer_train = y_fer_train.astype('int8')

X_fer_test_public, y_fer_test_public = np.rollaxis(data_fer[data_fer.Usage == "PublicTest"][["pixels", "emotion"]].values, -1)
X_fer_test_public = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_public]).reshape((-1, 48, 48))
y_fer_test_public = y_fer_test_public.astype('int8')

X_fer_test_private, y_fer_test_private = np.rollaxis(data_fer[data_fer.Usage == "PrivateTest"][["pixels", "emotion"]].values, -1)
X_fer_test_private = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_private]).reshape((-1, 48, 48))
y_fer_test_private = y_fer_test_private.astype('int8')


from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Conv2D, MaxPool2D, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


BATCH_SIZE=128

X_train = X_fer_train.reshape((-1, 48, 48, 1))
X_val = X_fer_test_public.reshape((-1, 48, 48, 1))
X_test = X_fer_test_private.reshape((-1, 48, 48, 1))
y_train = to_categorical(y_fer_train,7)
y_val = to_categorical(y_fer_test_public,7)
y_test = to_categorical(y_fer_test_private,7)

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
)

train_datagen.fit(X_train)
val_datagen.fit(X_train)

train_flow = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_flow = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
test_flow = val_datagen.flow(X_test, y_test, batch_size=1, shuffle=False)

DROPOUT_RATE = 0.3
CONV_ACTIVATION = "relu"

img_in = Input(shape=(48,48,1))

X = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(img_in)
X = BatchNormalization()(X)
X = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)

X = MaxPool2D((2, 2), strides=(2, 2), padding='same')(X)
X = Dropout(DROPOUT_RATE)(X)


X = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)

X = MaxPool2D((2, 2), strides=(2, 2), padding='same')(X)
X = Dropout(DROPOUT_RATE)(X)

X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)

X = MaxPool2D((2, 2), strides=(2, 2), padding='same')(X)
X = Dropout(DROPOUT_RATE)(X)

X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = MaxPool2D((2, 2), strides=(2, 2), padding='same')(X)
X = Dropout(DROPOUT_RATE)(X)

X = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation=CONV_ACTIVATION)(X)
X = BatchNormalization()(X)
X = MaxPool2D((2, 2), strides=(2, 2), padding='same')(X)
X = Dropout(DROPOUT_RATE)(X)

X = Flatten()(X)
X = Dense(2048, activation="relu")(X)
X = Dropout(DROPOUT_RATE)(X)
X = Dense(1024, activation="relu")(X)
X = Dropout(DROPOUT_RATE)(X)
X = Dense(512, activation="relu")(X)
X = Dropout(DROPOUT_RATE)(X)

out = Dense(7, activation='softmax')(X)

model = Model(inputs=img_in, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['categorical_accuracy'])

model.summary()

plot_model(model, show_shapes=True, show_layer_names=False)

early_stopping = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=20)
checkpoint_loss = ModelCheckpoint('best_loss_weights.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')
checkpoint_acc = ModelCheckpoint('best_accuracy_weights.h5', verbose=1, monitor='val_categorical_accuracy',save_best_only=True, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-7, cooldown=1, verbose=1)

history = model.fit_generator(
        train_flow, 
        steps_per_epoch= X_train.shape[0] // BATCH_SIZE,
        epochs=125, 
        validation_data=val_flow,
        validation_steps = X_val.shape[0] // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint_acc, checkpoint_loss, lr_reduce]
    )

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

def evaluate_model(weights_path):
    model.load_weights(weights_path)
    y_pred = model.predict_generator(test_flow, steps=X_test.shape[0])
    y_pred_cat = np.argmax(y_pred, axis=1)
    y_true_cat = np.argmax(test_flow.y, axis=1)
    report = classification_report(y_true_cat, y_pred_cat)
    print(report)

    conf = confusion_matrix(y_true_cat, y_pred_cat, normalize="true")

    labels = idx_to_emotion_fer.values()
    _, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(conf, annot=True, cmap='YlGnBu', 
                     xticklabels=labels, 
                     yticklabels=labels)

    plt.show()

evaluate_model('best_loss_weights.h5')

evaluate_model('best_accuracy_weights.h5')

!pip install -q tensorflow-model-optimization

import tensorflow_model_optimization as tfmot

model.load_weights('best_accuracy_weights.h5')

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Fine-tune prunned model on a couple of epochs,
# because the model may loose some of the learned features
pruning_epochs = 2
validation_split = X_val.shape[0] / X_train.shape[0]

num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) *125

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                             final_sparsity=0.8,
                                                             begin_step=0,
                                                             end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(loss='categorical_crossentropy', 
                          optimizer=Adam(0.001), 
                          metrics=['categorical_accuracy'])

print(model_for_pruning.summary())



import tempfile

logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit_generator(
    train_flow, 
    steps_per_epoch= X_train.shape[0] // BATCH_SIZE,
    epochs=pruning_epochs,
    validation_data=val_flow,
    validation_steps = X_val.shape[0] // BATCH_SIZE,
    callbacks=callbacks
)

model.save('pruned_model.h5')

evaluate_model('pruned_model.h5')

import os
import tensorflow as tf

compressed_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

tf.keras.models.save_model(compressed_model, 'compressed_model.h5', include_optimizer=False)

pruned_model_size = os.path.getsize('compressed_model.h5')
pruned_model_size_mb = pruned_model_size // 1024 // 1024

best_acc_model_size = os.path.getsize('best_accuracy_weights.h5')
best_acc_model_size_mb = best_acc_model_size // 1024 // 1024

improvement = int((1 - pruned_model_size / best_acc_model_size) * 100)

print(f'Pruned model size is {pruned_model_size_mb} Mbytes')
print(f'Pre-pruning model size is {best_acc_model_size_mb} Mbytes')
print(f'Improvement compared to pre-pruning model is {improvement}%')

import json

compressed_model_json = compressed_model.to_json()
with open('compressed_model.json', 'w') as f:
    f.write(compressed_model_json)

!pip install tensorflowjs

import tensorflowjs as tfjs

tfjs.converters.save_keras_model(compressed_model, 'compressed_model_js.json')
model.save_weights("compressed_model.h5")

