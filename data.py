#from load_duf import DataloadFromFolder
from load_train import DataloadFromFolder
from load_test import DataloadFromFolderTest
from load_test import DataloadFromFolderTest2
from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
             ToTensor(),
            ])

def get_training_set(data_dir, upscale_factor, data_augmentation, file_list):
    return DataloadFromFolder(data_dir, upscale_factor, data_augmentation, file_list, transform=transform())

def get_test_set(data_dir, upscale_factor, file_list):
    return DataloadFromFolderTest(data_dir, upscale_factor, file_list,transform=transform())

def get_test_set2(data_dir, upscale_factor, file_list):
    return DataloadFromFolderTest2(data_dir, upscale_factor, file_list, transform=transform())


