from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image, ImageFile
from pathlib import Path
from itertools import chain


class PISCDataset(Dataset):
    def __init__(self, path):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.path = Path(path).absolute()
        self.transforms = self.get_transforms()
        self.classes = ['friends', 'family', 'couple', 'professional', 'commercial', 'no_relation']
        self.dir = [list((self.path / c).iterdir()) for c in self.classes]
        self.dir = sorted(chain(*self.dir))

    def __getitem__(self, idx):
        relation = str(self.dir[idx].parent).split('/')[-1]
        img_path = [str(path) for path in self.dir[idx].glob('*.png')]

        person1 = Image.open(img_path[0]).convert('RGB')
        person2 = Image.open(img_path[1]).convert('RGB')

        if self.transforms is not None:
            person1 = self.transforms(person1)
            person2 = self.transforms(person2)
            
        return person1, person2, self.classes.index(str(relation))

    def __len__(self):
        return len(self.dir)

    def get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform
