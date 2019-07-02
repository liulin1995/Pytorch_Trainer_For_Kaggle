from torch.utils.data import DataLoader
from .ImageDataset import ImageDataset
from .ImageTransform import image_transform, image_transform_val
import sys
sys.path.append('../')
from Data_Config import data_config

image_train_dataset = ImageDataset(data_config.IMG_TRAIN_PATH, transform=image_transform)
image_val_dataset = ImageDataset(data_config.IMG_VAL_PATH, transform=image_transform_val)
image_test_dataset = ImageDataset(data_config.IMG_TEST_PATH, train_mode=False, transform=image_transform_val)
vis_dataset = ImageDataset(data_config.IMG_TRAIN_PATH, transform=vis_transform)

train_loader = DataLoader(image_train_dataset, batch_size=data_config.TRAIN_BATCH_SIZE, shuffle=data_config.SHUFFLE,
                          num_workers=data_config.NUM_WORKERS, pin_memory=True)
train_eval_loader = DataLoader(image_train_dataset, batch_size=data_config.VAL_BATCH_SIZE, shuffle=data_config.SHUFFLE,
                          num_workers=data_config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(image_val_dataset, batch_size=data_config.VAL_BATCH_SIZE, shuffle=data_config.SHUFFLE,
                        num_workers=data_config.NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(image_test_dataset, batch_size=data_config.TEST_BATCH_SIZE,
