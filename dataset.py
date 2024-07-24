from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random
import pandas as pd
from tqdm import tqdm


class JustRAIGSDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, stage, transforms=None):
        """
        :param data_folder: # folder with JSON data files
        :param stage: one of 'train' or 'test'
        """

        self.data_folder = data_folder
        self.stage = stage.lower()

        assert self.stage in {'train', 'test'}
        self.transforms = transforms

        # Read list of image-paths
        if self.stage == 'train':
            with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder, 'test_images.json'), 'r') as j:
                self.images = json.load(j)
        self.load_ground_truths()
        true_gt = [self.ground_truths[os.path.splitext(os.path.basename(image))[0]] for image in self.images].count(1)
        false_gt = len(self.images) - true_gt
        print(f'Loaded {len(self.images)} {self.stage} images\tTrue: {true_gt}\tFalse: {false_gt}')

    def load_ground_truths(self):
        df = pd.read_csv(os.path.join(
            self.data_folder, 'JustRAIGS_Train_labels.csv'), sep=';')

        df['GT'] = (df['Final Label'] == 'RG').astype(int)
        self.ground_truths = df.set_index('Eye ID')['GT'].to_dict()

    def __getitem__(self, i):
        img = Image.open(os.path.join(
            self.data_folder, self.images[i]), mode='r')
        img = img.convert('RGB')

        index = os.path.splitext(os.path.basename(self.images[i]))[0]

        gt = torch.tensor(self.ground_truths[index])

        if self.transforms:
            img = self.transforms(img)

        return img, gt

    def __len__(self):
        return len(self.images)


def prepare_data(resize=None):
    data_folder = '/Users/youssefshaarawy/Documents/Datasets/JustRAIGS'
    folders = sorted([os.path.join(data_folder, x) for x in os.listdir(data_folder)
                     if os.path.isdir(os.path.join(data_folder, x))])
    df = pd.read_csv(os.path.join(
        data_folder, 'JustRAIGS_Train_labels.csv'), sep=';', index_col='Eye ID')
    true_images, false_images = [], []
    for folder in folders:
        for img in os.listdir(folder):
            if img.endswith(".jpg") or img.endswith(".JPG"):
                image_name = os.path.splitext(img)[0]
                if df.loc[image_name, 'Final Label'] == 'RG':
                    true_images.append(os.path.join(
                        os.path.basename(folder), img))
                else:
                    false_images.append(os.path.join(
                        os.path.basename(folder), img))

    random.seed(43)
    random.shuffle(true_images)
    random.shuffle(false_images)

    split = 0.8
    true_train_size, false_train_size = int(
        len(true_images) * split), int(len(false_images) * split)
    train_paths = [*true_images[:true_train_size],
                   *false_images[:false_train_size]]
    test_paths = [*true_images[true_train_size:],
                  *false_images[false_train_size:]]

    random.shuffle(train_paths)
    random.shuffle(test_paths)

    print(len(true_images))
    print(len(false_images))

    with open(os.path.join(data_folder, 'train_images.json'), 'w') as j:
        json.dump(train_paths, j)
    with open(os.path.join(data_folder, 'test_images.json'), 'w') as j:
        json.dump(test_paths, j)

    if resize:
        def resize_image(img_path, resize):
            try:
                with Image.open(img_path) as im:
                    im = im.resize((resize, resize))
                    im.save(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        def process_folder(folder, resize):
            images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith((".jpg", ".JPG"))]
            
            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(lambda img: resize_image(img, resize), images), total=len(images), desc=f"Processing {folder}"))

        def resize_images_in_folders(folders, resize):
            for folder in folders:
                process_folder(folder, resize)

        resize_images_in_folders(folders, resize)


if __name__ == '__main__':
    prepare_data(512)
