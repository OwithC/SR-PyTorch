from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path


def get_bbox(id):
    with open('/PATH/PISC/annotation_image_info.json') as j:
        img_info = json.load(j)

    for info in img_info:
        if info['id'] == id:
            return info['bbox']


if __name__ == '__main__':
    data_path = Path('./image').absolute()
    save_dir = data_path.parent / 'crop'

    if not save_dir.is_dir():
        Path.mkdir(save_dir)

    classes = ['friends', 'family', 'couple', 'professional', 'commercial', 'no_relation']
    c_idx = [0] * len(classes)
    for c in classes:
        class_dir = save_dir / c
        if not class_dir.is_dir():
            Path.mkdir(class_dir)

    with open('/PATH/PISC/relationship.json') as j:
        relationship = json.load(j)

    relationship = sorted(relationship.items())

    for i, r in enumerate(tqdm(relationship)):
        id, relation = r
        img_path = data_path / f'{id.zfill(5)}.jpg'

        if not img_path.is_file():
            continue

        person = []
        img = Image.open(img_path)
        bboxes = get_bbox(int(id))
        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            person.append(img.crop((x, y, x + w, y + h)))

        for pair, label in relation.items():
            a, b = pair.split()
            a, b = int(a), int(b)
            person1, person2 = person[a-1], person[b-1]
            c_idx[label-1] += 1
            save_path = save_dir / classes[label-1] / str(f'%05d' % c_idx[label-1])

            if not save_path.is_dir():
                Path.mkdir(save_path)

            person1.save(save_path / '1.png')
            person2.save(save_path / '2.png')
