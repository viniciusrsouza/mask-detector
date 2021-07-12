import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import (Dense, Flatten, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from mask_detection.data_provider import DataProvider


class MaskDetector():
    INIT_LR = 1e-2
    EPOCHS = 5
    BS = 32

    def __init__(self, dataset=None, augment=None):
        (self.y_train, self.y_test, self.x_train,
         self.x_test) = dataset or ([], [], [], [])
        self.model = keras.Sequential()
        self.strategy = tf.distribute.get_strategy()
        self.classes = ["with_mask", "without_mask"]
        self.augment = augment

    def preprocess_data(self):
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        conditions = [
            self.y_train == self.classes[0],
            self.y_train == self.classes[1]]
        choices = [0, 1]

        self.y_train = np.select(conditions, choices)

    def build_model(self):
        baseModel = MobileNet(weights="imagenet", include_top=False,
                              input_tensor=Input(shape=(244, 244, 3)))

        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        self.model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(learning_rate=MaskDetector.INIT_LR,
                   decay=MaskDetector.INIT_LR / MaskDetector.EPOCHS)
        self.model.compile(loss="binary_crossentropy", optimizer=opt,
                           metrics=["accuracy"])
        return self.model

    def train_model(self):
        H = self.model.fit(
            self.augment.flow(self.x_train, self.y_train,
                              batch_size=MaskDetector.BS),
            steps_per_epoch=len(self.x_train) // MaskDetector.BS,
            validation_data=(self.x_test, self.y_test),
            validation_steps=len(self.x_test) // MaskDetector.BS,
            epochs=MaskDetector.EPOCHS)
        return self

    def evaluate(self):
        y_pred = self.model.predict(self.x_test, batch_size=MaskDetector.BS)
        y_pred = np.argmax(y_pred, axis=1)

        print(classification_report(self.y_test.argmax(axis=1),
                                    y_pred,
                                    target_names=['with_mask', 'without_mask']))

    def predict(self, img):
        image = img_to_array(img)
        image = preprocess_input(image)
        image = np.array([image], dtype="float32")

        y_pred = self.model.predict(image)[0]

        max_index = np.argmax(y_pred)
        result = (self.classes[max_index], y_pred[max_index])
        print('result', result, end='\t')
        print('y', y_pred)
        return result


if __name__ == '__main__':
    provider = DataProvider() \
        .load_dataset_as_array() \
        .split_dataset() \
        .augment_data()

    dataset = provider.get_splitted_dataset()

    mask_detector = MaskDetector(dataset, provider.augment)
    # mask_detector.preprocess_data()
    mask_detector.build_model()
    mask_detector.train_model()
    mask_detector.model.save('output/')

    mask_detector.model = keras.models.load_model('output')
    mask_detector.evaluate()
