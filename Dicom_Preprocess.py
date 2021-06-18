from Utils_folder.Utils_Preprocess import *
from Utils_folder.HN_Constants import ALL_IM_SIZE, ROI_ORDER, DICOM_DIR
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=True)
    parser.add_argument('--filelist', type=bool, default=False)
    parser.add_argument('--size', type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists('Filelist'):
        os.makedirs('Filelist')
    if args.preprocess:
        print('Resize to ',ALL_IM_SIZE)
        flip = False  #switch for enable flip for image for horizontal plane
        print('Flip =',flip)
        preprocess(ALL_IM_SIZE, DICOM_DIR, flip, data_path = 'Data')

    if args.filelist:
        print('Generate file list')
        train_filelist('Data', ROI_ORDER)
        print('Done')
    
    if args.size:
        print('Generate size and ratio')
        train_file_lists = pd.read_csv('Filelist/Train_Filelist_ROI.csv')
        size_ratio(train_file_lists, ROI_ORDER)
        print('Done')