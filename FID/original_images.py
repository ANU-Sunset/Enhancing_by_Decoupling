import os
import random
import glob
import shutil

if __name__ == '__main__':
    data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root)]
    print(f'images: {len(image_list)}')
    train_size=20000
    train_list = image_list[:train_size]
    print(f"train_images size: {train_size}")
    assert len(train_list) > 0

    ## validation
    val_size = 2000
    print(f"validation_images size: {val_size}")
    val_list = image_list[train_size:train_size+val_size]

    select_train = random.sample(train_list, 200)
    # print(select_train)

    dst_dir = '/scratch/u7076589/engn8536/Datasets/z_map/FID/celebA256x256x200'

    for jpgfile in select_train:
        jpgfile = os.path.join(data_root, jpgfile)
        shutil.copy(jpgfile, dst_dir)