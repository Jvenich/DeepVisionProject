import pandas as pd
from shutil import move
import os

df = pd.read_csv('./datasets/artset/train_info.csv')
all_images = os.listdir('./datasets/artset/train')

co = 0

for image in all_images:
    print(image)

    class_label = df[df['filename'] == image]['style']
    class_label = str(list(class_label)[0])

    if not os.path.exists(os.path.join('./datasets/artset', class_label)):
        os.mkdir(os.path.join('./datasets/artset', class_label))

    path_from = os.path.join('./datasets/artset/train', image)
    path_to = os.path.join('./datasets/artset', class_label, image)

    move(path_from, path_to)
    print('Moved {} to {}'.format(image, path_to))
    co += 1

print('Moved {} images.'.format(co))