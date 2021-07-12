import collections
import os
import shutil
from pathlib import Path
from random import shuffle

import numpy as np
import xmltodict
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataProvider():
    def __init__(self):
        self._annotations_dir = os.listdir('data/annotations')
        self.y_train = []
        self.y_test = []
        self.x_train = []
        self.x_test = []
        self.labeled_dataset = []
        self.x_data = []
        self.y_data = []
        self.augment = None

    def _get_rect_and_label_from_object(self, obj):
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        rect = (xmin, ymin, xmax, ymax)
        label = obj['name']
        return (rect, label)

    def gen_labeled_dataset(self):
        with_mask_path = Path('data/with_mask')
        without_mask_path = Path('data/without_mask')
        mask_weared_incorrect_path = Path('data/mask_weared_incorrect')
        if with_mask_path.exists():
            shutil.rmtree(with_mask_path)
        if without_mask_path.exists():
            shutil.rmtree(without_mask_path)
        if mask_weared_incorrect_path.exists():
            shutil.rmtree(mask_weared_incorrect_path)

        result = []
        progress_list = tqdm(self._annotations_dir,
                             desc='generating labeled dataset',
                             colour='green')
        for f_label in progress_list:
            f_image = f_label.replace('xml', 'png')
            f_image = f'data/images/{f_image}'
            im = Image.open(f_image)

            f_label = f'data/annotations/{f_label}'
            lb = open(f_label, 'r')
            labels = xmltodict.parse(lb.read())['annotation']

            if isinstance(labels['object'], list):
                for obj in labels['object']:
                    (rect, label) = self._get_rect_and_label_from_object(obj)
                    res_img = im.crop(rect)
                    res_img = res_img.resize((244, 244), Image.ANTIALIAS)
                    result.append((res_img, label))
            else:
                obj = labels['object']
                (rect, label) = self._get_rect_and_label_from_object(obj)
                res_img = im.crop(rect)
                res_img = res_img.resize((244, 244), Image.ANTIALIAS)
                result.append((res_img, label))
        self.labeled_dataset = result
        return self

    def save_labeled_dataset(self):
        Path('data/with_mask').mkdir(parents=True, exist_ok=True)
        Path('data/without_mask').mkdir(parents=True, exist_ok=True)
        Path('data/mask_weared_incorrect').mkdir(parents=True, exist_ok=True)

        dataset_with_range = list(zip(self.labeled_dataset,
                                      range(len(self.labeled_dataset))))
        progress = tqdm(dataset_with_range,
                        desc='saving cropped images',
                        colour='green')
        for ((img, label), index) in progress:
            img.save(f'data/{label}/{index}.png')
        return self

    def load_dataset_as_array(self):
        with_mask_path = Path('data/with_mask')
        without_mask_path = Path('data/without_mask')
        mask_weared_incorrect_path = Path('data/mask_weared_incorrect')
        if not (with_mask_path.exists() and without_mask_path.exists() and mask_weared_incorrect_path.exists()):
            raise Exception('dataset paths do not exist.')

        progress = tqdm(list(with_mask_path.iterdir()),
                        desc='loading \'with_mask\': ',
                        colour='green')
        for file in progress:
            image = load_img(file, target_size=(244, 244))
            image = img_to_array(image)
            image = preprocess_input(image)
            self.x_data.append(image)
            self.y_data.append('with_mask')

        progress = tqdm(list(without_mask_path.iterdir()),
                        desc='loading \'without_mask\': ',
                        colour='green')
        for file in progress:
            image = load_img(file, target_size=(244, 244))
            image = img_to_array(image)
            image = preprocess_input(image)
            self.x_data.append(image)
            self.y_data.append('without_mask')

        progress = tqdm(list(mask_weared_incorrect_path.iterdir()),
                        desc='loading \'mask_weared_incorrect\': ',
                        colour='green')
        for file in progress:
            image = load_img(file, target_size=(244, 244))
            image = img_to_array(image)
            image = preprocess_input(image)
            self.x_data.append(image)
            self.y_data.append('without_mask')

        return self

    def split_dataset(self, ratio=0.8):
        lb = LabelBinarizer()
        labels = lb.fit_transform(self.y_data)
        labels = to_categorical(labels)

        data = np.asarray(self.x_data, dtype="float32")
        labels = np.asarray(labels)

        print(data.shape)
        print(labels.shape)
        print(labels[0])

        (self.x_train,
         self.x_test,
         self.y_train,
         self.y_test) = train_test_split(data,
                                         labels,
                                         test_size=1-ratio,
                                         stratify=labels,
                                         random_state=42)
        return self

    def get_splitted_dataset(self):
        return (self.y_train, self.y_test, self.x_train, self.x_test)

    def augment_data(self):
        self.augment = ImageDataGenerator(rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode="nearest")
        return self


if __name__ == "__main__":
    d = DataProvider()
    d.gen_labeled_dataset()
    d.save_labeled_dataset()
    (y_train, y_test, x_train, x_test) = d.get_splitted_dataset()
