import torch.utils.data as data
import torchvision.transforms as transforms
import datasets
import warnings

warnings.filterwarnings('ignore')

class ImageNetDataset(data.Dataset):
    def __init__(self, split = 'train'):
        dataset = datasets.load_dataset('ILSVRC/imagenet-1k', cache_dir = '/storage/vsub851/.cache')
        self.data = dataset[split]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        img = entry['image'].convert('RGB')
        label = entry['label']

        img = self.transform(img)
        return {'image': img, 'label': label}