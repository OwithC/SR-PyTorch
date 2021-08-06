import os.path

from PIL import Image
import json
import glob
import os
from tqdm import tqdm


def get_bbox(id):
    with open('/PATH/PISC/annotation_image_info.json') as j:
        img_info = json.load(j)

    for info in img_info:
        if info['id'] == id:
            return info['bbox']


def get_file(img_path, types):
    file_dic = {}
    for type in types:
        file_dic[type] = glob.glob(os.path.join(img_path, type) + '/*.jpg')

    return file_dic


if __name__ == '__main__':
    img_path = '/PISC/IMAGE/PATH/'
    types = ['train', 'test']

    files = get_file(img_path, types)

    save_dir = os.path.join(img_path, 'preprocessed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, file in tqdm(enumerate(files.values())):
        type_dir = os.path.join(save_dir, types[i])
        if not os.path.exists(type_dir):
            os.makedirs(type_dir)

        for img_path in file:
            id = img_path.split('/')[-1].split('.')[0]
            img_dir = os.path.join(type_dir, id)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            bboxes = get_bbox(int(id))

            img = Image.open(img_path)
            for j, bbox in enumerate(bboxes):
                save_path = os.path.join(img_dir, str(j)) + '.png'
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                crop_img = img.crop((x, y, x + w, y + h))
                crop_img.save(save_path)
