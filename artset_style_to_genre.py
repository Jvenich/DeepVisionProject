import pandas as pd
from shutil import copyfile
import os

df = pd.read_csv('./train_info.csv')
all_folders = os.listdir('./datasets/artset/')

if not os.path.exists('./datasets/artset_genre'):
    os.mkdir('./datasets/artset_genre')

co = 0

not_copied = []
empty_label = []

for folder in all_folders:
    all_images = os.listdir(os.path.join('./datasets/artset', folder))
    for image in all_images:
        print(image)

        class_label = df[df['filename'] == image]['genre']
        
        if len(list(class_label)) == 0:
            print("{} has empty class label".format(image))
            empty_label.append(image)
            continue
       
        try:
            class_label = str(list(class_label)[0])
        except:
            print("{} could not be copied.".format(image))
            not_copied.append(image)
            continue

        if not os.path.exists(os.path.join('./datasets/artset_genre', class_label)):
            os.mkdir(os.path.join('./datasets/artset_genre', class_label))

        path_from = os.path.join('./datasets/artset', folder, image)
        path_to = os.path.join('./datasets/artset_genre', class_label, image)
        
        copyfile(path_from, path_to)
        print('Copy {} to {}'.format(image, path_to))
        co += 1

print('Copied {} images.'.format(co))
print('{} images not copied, because of empty label'.format(len(empty_label)))
print('Additional {} images not copied, because of unknown reason'.format(len(not_copied)))
print("list of images with empty labels:")
print(empty_label)
print('other images that were not copied:')
print(not_copied)


