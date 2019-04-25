import pandas as pd
import os
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
TRAIN_IMAGE_2000_LIST = "./ChestX-ray14/labels/train_list_2000.txt"
TRAIN_IMAGE_10000_LIST = "./ChestX-ray14/labels/train_list_10000.txt"
DEV_IMAGE_LIST = './ChestX-ray14/labels/val_list.txt'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
DIR_BASE = "./ChestX-ray14/labels/"

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def main():
    train_df = pd.read_table(TRAIN_IMAGE_LIST, sep=' ', header=None)
    train_2000_df = pd.read_table(TRAIN_IMAGE_2000_LIST, sep=' ', header=None)
    train_10000_df = pd.read_table(
        TRAIN_IMAGE_10000_LIST, sep=' ', header=None)
    dev_df = pd.read_table(DEV_IMAGE_LIST, sep=' ', header=None)
    test_df = pd.read_table(TEST_IMAGE_LIST, sep=' ', header=None)
    train_df.columns = ["filename"]+CLASS_NAMES
    train_2000_df.columns = ["filename"]+CLASS_NAMES
    train_10000_df.columns = ["filename"]+CLASS_NAMES
    dev_df.columns = ["filename"]+CLASS_NAMES
    test_df.columns = ["filename"]+CLASS_NAMES
    train_df.to_hdf(os.path.join(DIR_BASE, "train_df.h5"), key="key")
    train_2000_df.to_hdf(os.path.join(DIR_BASE, "train_2000_df.h5"), key="key")
    train_10000_df.to_hdf(os.path.join(
        DIR_BASE, "train_10000_df.h5"), key="key")
    dev_df.to_hdf(os.path.join(DIR_BASE, "dev_df.h5"), key="key")
    test_df.to_hdf(os.path.join(DIR_BASE, "test_df.h5"), key="key")
    print(train_df.iloc[:, 1:].mean())
    print(train_2000_df.iloc[:, 1:].mean())
    print(train_10000_df.iloc[:, 1:].mean())
    print(dev_df.iloc[:, 1:].mean())
    print(test_df.iloc[:, 1:].mean())


if __name__ == "__main__":
    main()
