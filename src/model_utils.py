import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f" — val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall _val_recall")
        return


metrics = Metrics()

def create_and_train_model(train_path, val_path):
    """Create and train a deep learning model."""
    img_height, img_width = 150, 150
    batch_size = 32
    epochs = 20

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
    )

    val_generator = train_datagen.flow_from_directory(
        val_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(filters=32, padding="same", kernel_size=(3, 3), activation="relu", input_shape=(img_height, img_width, 1)),
        MaxPooling2D(pool_size=(2, 2),padding='same'),
        Dropout(0.3),
        Conv2D(filters=64, padding="same", kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2),padding='same'),
        Dropout(0.3),
        Conv2D(filters=128, padding="same", kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2),padding='same'),
        Dropout(0.3),
        Flatten(),
        Dense(units=256, activation="relu"),
        Dropout(0.5),
        Dense(units=512, activation="relu"),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    TRAIN_STEPS_PER_EPOCH = int(np.ceil((train_generator.samples*0.8/batch_size)-1))
    VAL_STEPS_PER_EPOCH = int(np.ceil((val_generator.samples*0.2/batch_size)-1))

    history = model.fit(
        train_generator,
        steps_per_epoch= TRAIN_STEPS_PER_EPOCH,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps = VAL_STEPS_PER_EPOCH,
        callbacks=[early_stopping]
    )

    return model, history

def save_model(model, model_path, model_name="LungDiseaseModel-CNN.keras"):
    """Save the trained model."""
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    full_model_path = os.path.join(model_path, model_name)
    model.save(full_model_path)
    return full_model_path

def load_and_evaluate_model(model_path, val_path, class_names, img_height=150, img_width=150):
    """Load saved model and evaluate on validation data."""
    model_weight = load_model(model_path)
    predictions = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(val_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = image.load_img(img_path, color_mode='grayscale',target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            #print(img_array.shape)
            img_array = np.expand_dims(img_array, axis=0)
            # print(img_array.shape)
            img_array /= 255.0
            prediction = model_weight.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction)
            predictions.append(predicted_class)
            labels.append(class_names.index(class_name))

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
