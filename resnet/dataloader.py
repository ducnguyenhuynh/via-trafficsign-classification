import os 
import torch
import copy
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Dataloader():
    def __init__(self, data_path, image_size):
        
        self.train_dir = os.path.join(data_path, 'train')
        self.test_dir = os.path.join(data_path, 'test')
        self.image_size = image_size

    
    
    def get_dataset(self):
        def calculate_means_stds():
            train_data = datasets.ImageFolder(root = self.train_dir, 
                                    transform = transforms.ToTensor())
            means = torch.zeros(3)
            stds = torch.zeros(3)

            for img, label in train_data:
                means += torch.mean(img, dim = (1,2))
                stds += torch.std(img, dim = (1,2))
            means /= len(train_data)
            stds /= len(train_data)

            return list(means), list(stds)

        means, stds = calculate_means_stds()

        # augmentation, normalize
        train_transforms =  transforms.Compose([
                                            transforms.Resize(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = means, std = stds),
                                                ])
        
        test_transforms =  transforms.Compose([
                                            transforms.Resize(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = means, std = stds)
                                                ])
        
        
        train_data = datasets.ImageFolder(root = self.train_dir, transform = train_transforms)
        test_data = datasets.ImageFolder(root = self.test_dir, transform = test_transforms)

        # create valid data
        VALID_RATIO = 0.9

        n_train_examples = int(len(train_data) * VALID_RATIO)
        n_valid_examples = len(train_data) - n_train_examples

        train_data, valid_data = data.random_split(train_data, 
                                                [n_train_examples, n_valid_examples])
        
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms
        
        return train_data, valid_data, test_data
