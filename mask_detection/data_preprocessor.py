import os
import shutil
from pathlib import Path
from random import shuffle

import xmltodict
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


class DataPreProcessor():
    def __init__(self):
        self._annotations_dir = os.listdir('data/annotations')

    def get_rect_and_label_from_object(self, obj):
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
                    (rect, label) = self.get_rect_and_label_from_object(obj)
                    res_img = im.crop(rect)
                    res_img = res_img.resize((32, 32), Image.ANTIALIAS)
                    result.append((res_img, label))
            else:
                obj = labels['object']
                (rect, label) = self.get_rect_and_label_from_object(obj)
                res_img = im.crop(rect)
                res_img = res_img.resize((32, 32), Image.ANTIALIAS)
                result.append((res_img, label))
        return result

    def save_labeled_dataset(self, dataset):
        Path('data/with_mask').mkdir(parents=True, exist_ok=True)
        Path('data/without_mask').mkdir(parents=True, exist_ok=True)
        Path('data/mask_weared_incorrect').mkdir(parents=True, exist_ok=True)

        dataset_with_range = list(zip(dataset, range(len(dataset))))
        progress = tqdm(dataset_with_range,
                        desc='saving cropped images',
                        colour='green')
        for ((img, label), index) in progress:
            img.save(f'data/{label}/{index}.png')

    def get_file_index(self):
        print('a')

    def get_labeled_dataset(self):
        with_mask_path = Path('data/with_mask')
        without_mask_path = Path('data/without_mask')
        mask_weared_incorrect_path = Path('data/mask_weared_incorrect')
        if not (with_mask_path.exists() and without_mask_path.exists() and mask_weared_incorrect_path.exists()):
            raise Exception('dataset paths do not exist.')

        res = []
        progress = tqdm(list(with_mask_path.iterdir()),
                        desc='loading \'with_mask\': ',
                        colour='green')
        for file in progress:
            img = load_img(file, target_size=(32, 32))
            img = img_to_array(img)
            res.append((img, 'with_mask'))

        progress = tqdm(list(without_mask_path.iterdir()),
                        desc='loading \'without_mask\': ',
                        colour='green')
        for file in progress:
            img = load_img(file, target_size=(32, 32))
            img = img_to_array(img)
            res.append((img, 'without_mask'))

        progress = tqdm(list(mask_weared_incorrect_path.iterdir()),
                        desc='loading \'mask_weared_incorrect\': ',
                        colour='green')
        for file in progress:
            img = load_img(file, target_size=(32, 32))
            img = img_to_array(img)
            res.append((img, 'mask_weared_incorrect'))
        return res

    def split_labeled_dataset(self, labeled_dataset, ratio=0.8):
        shuffle(labeled_dataset)
        y_train = []
        y_test = []
        x_train = []
        x_test = []
        i = 0
        limit = int(len(labeled_dataset) * ratio)
        progress = tqdm(labeled_dataset,
                        desc='splitting dataset',
                        colour='green')
        for (img, label) in progress:
            if i < limit:
                x_train.append(img)
                y_train.append(label)
            else:
                x_test.append(img)
                y_test.append(label)
            i += 1
        return (y_train, y_test, x_train, x_test)


if __name__ == "__main__":
    d = DataPreProcessor()
    res = d.gen_labeled_dataset()
    d.save_labeled_dataset(res)
    res = d.get_labeled_dataset()
    d.split_labeled_dataset(res)
