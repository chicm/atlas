import os
import local_settings

DATA_DIR = local_settings.DATA_DIR

MODEL_DIR = os.path.join(DATA_DIR, 'models')

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_LABEL = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'sample_submission.csv')
