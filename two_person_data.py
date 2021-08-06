import torch
from torchvision import transforms

from glob import glob
from PIL import Image, ImageFile
import json
import os


class PISCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, id_list):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.path = data_path
        self.transforms = self.get_transforms()
        self.img_idx = -1
        self.item = []

        # 전체 relation 얻기
        # 경로 수정 필요
        with open('/PATH/PISC/relationship.json') as j:
            self.relationship = json.load(j)

        self.ids = id_list
        self.relations = [self.relationship[id] for id in self.ids]
        self.length = sum([len(r) for r in self.relations])

    def __getitem__(self, idx):
        item = self.get_item()
        return item

    def __len__(self):
        return self.length

    def get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transform

    def get_item(self):
        '''

        :return:
        두 사람의 각 영역에 대한 이미지, 두 사람의 관계에 대한 라벨
        '''
        if not self.item:
            self.img_idx += 1

            # 하나의 이미지에서 잘라서 저장한 사람 영역의 이미지를 리스트에 저장
            people = []

            paths = glob(os.path.join(self.path + self.ids[self.img_idx].zfill(5)) + '/*.png')
            paths.sort()

            for path in paths:
                people.append(Image.open(path).convert('RGB'))

            # 주석에 따라 두 사람의 이미지와 관계를 리스트에 저장
            self.item = []
            for i, (x, y) in enumerate(self.relations[self.img_idx].items()):
                a, b = x.split()
                a, b = int(a), int(b)
                person1 = people[a - 1]
                person2 = people[b - 1]
                if self.transforms is not None:
                    person1 = self.transforms(person1)
                    person2 = self.transforms(person2)

                self.item.append([person1, person2, y-1])

        if self.img_idx >= len(self.relations) - 1:
            self.img_idx = -1

        return self.item.pop(0)
