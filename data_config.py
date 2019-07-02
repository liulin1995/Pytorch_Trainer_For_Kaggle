class DataConfig:
    IMG_TRAIN_PATH = 'C:/cvpr2019_kaggle_competition/train'
    IMG_VAL_PATH = 'F:/cvpr2019_kaggle_competition/val'
    IMG_TEST_PATH = 'F:/cvpr2019_kaggle_competition/test'
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    SHUFFLE = True
    NUM_WORKERS = 4


data_config = DataConfig()