import pandas as pd
from shutil import copyfile
import os

df = pd.read_csv('./train_info.csv')
all_folders = os.listdir('./datasets/artset/')

if not os.path.exists('./datasets/artset_style'):
    os.mkdir('./datasets/artset_style')

co = 0

for folder in all_folders:
    all_images = os.listdir(os.path.join('./datasets/artset', folder))
    for image in all_images:
        print(image)

        class_label = df[df['filename'] == image]['genre']
        class_label = str(list(class_label)[0])

        if not os.path.exists(os.path.join('./datasets/artset_style', class_label)):
            os.mkdir(os.path.join('./datasets/artset_style', class_label))

        path_from = os.path.join('./datasets/artset', folder, image)
        path_to = os.path.join('./datasets/artset_style', class_label, image)

        copyfile(path_from, path_to)
        print('Copy {} to {}'.format(image, path_to))
        co += 1

print('Copied {} images.'.format(co))

