import os, sys, torch, logging, shutil, h5py, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from torch import nn
from Utils_folder.Utils import *
from Utils_folder.HN_Constants import *
from Model.Model_First import AttU_Net_3layer
from Model.Model_Second import AttU_Net_4layer
from Model.Model_Old import UNet3D
from Utils_folder.Dataset_Util import Test_Dataset
from tqdm import tqdm
import pandas as pd
from Utils_folder.Utils_Preprocess import *

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

THRESHOLD = False
BENCHMARK_DICOM = '/media/lingshu/data2/HNProsp/'
BENCHMARK_IMAGE = 'Benchmark_Image'
OUTPUT_DIR = 'Benchmark_Output'
LABELMAP_DIR = 'Benchmark_Labelmaps'
CHECKPOINT_DIR = 'Checkpoint'
OLD_CHECKPOINT_DIR = 'checkpoint_3layer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_OLD_MODEL = True
if LOAD_OLD_MODEL:
    STEP1_FEATURE_ROOT = 48
    DROPOUT_RATIO = 0.3

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(file_name, output, read_info, roi, stage, adaptive_threshold):
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    output_file = os.path.join(OUTPUT_DIR, roi, roi + '_'+ stage + '_' + os.path.basename(file_name) )
    f_h5 = h5py.File(output_file, 'w')
    output = output.cpu()
    output = output.squeeze()
    slice_prob = output
    probs = slice_prob.numpy()
    slice_prediction = np.zeros(probs.shape, dtype=np.int8)
    if stage == 'first':
        threshold = FIRST_THRESHOLD[roi] if adaptive_threshold else 0.5
        slice_prediction[probs < threshold] = 0
        slice_prediction[probs >= threshold] = 1
        prediction = restore_labels(slice_prediction, roi, read_info, stage)
    else:
        probs = restore_labels(probs, roi, read_info, stage)
        prediction = np.round(probs)

    f_h5.create_dataset('probs', data=probs, compression="gzip", compression_opts=9)
    f_h5.create_dataset('predictions', data=prediction, compression="gzip", compression_opts=9)
    f_h5.close()

def test_model(test_dataloader, net, path, roi, stage, adaptive_threshold = False):
     with torch.no_grad():
          checkpoint = torch.load(path)
          net.load_state_dict(checkpoint['net'])
          net.eval()
          with tqdm(test_dataloader) as t:
               for data in t:
                    image = data[0].to(device)
                    read_info = data[1]
                    file_name = data[2][0]
                    output = net(image)
                    t.set_description("File: %s"%file_name)
                    test(file_name, output, read_info, roi, stage, adaptive_threshold)

def main(FIRST_STAGE, SECOND_STAGE, BINARY_MAPS, rois):

     if not os.path.exists(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR)

     testing_paths =[name for name in os.listdir(os.path.join(BENCHMARK_IMAGE,'Images')) if '.hdf5' in name]

     for roi in rois:
          index = ROI_ORDER.index(roi)
          if FIRST_STAGE:
               stage = 'first'
               if not os.path.exists(os.path.join(OUTPUT_DIR, roi)):
                    os.makedirs(os.path.join(OUTPUT_DIR, roi))
               if LOAD_OLD_MODEL:
                    Unet_first = UNet3D(in_channels = 1, out_channels = 1, final_sigmoid = True, f_maps = 48, layer_order='crg', num_groups = 8, num_levels = 3, is_segmentation = True, testing = True)
               else:
                    Unet_first = AttU_Net_3layer(feature = STEP1_FEATURE_ROOT, P = DROPOUT_RATIO, activate = True)
               model_path = os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP1_FEATURE_ROOT) +'_first.pt')
               if torch.cuda.device_count() > 1 and not device.type == 'cpu':
                    Unet_first = nn.DataParallel(Unet_first)
                    logging.warning(f'Using {torch.cuda.device_count()} GPUs for testing First stage {roi}')
               Unet_first = Unet_first.to(device)
               test_dataset = Test_Dataset(testing_paths, BENCHMARK_IMAGE, OUTPUT_DIR, roi, ALL_IM_SIZE, stage)
               test_loader = DataLoaderX(test_dataset, batch_size=1, shuffle = False)
               test_model(test_loader, Unet_first, model_path, roi, stage, adaptive_threshold = False)

          im_size = tuple([int(x/8)*8 for x in IM_SIZES[index]])
          if SECOND_STAGE:
               stage = 'second'
               Unet = AttU_Net_4layer(feature = STEP2_FEATURE_ROOT, P = DROPOUT_RATIO, activate = True)
               if torch.cuda.device_count() > 1 and not device.type == 'cpu':
                    Unet = nn.DataParallel(Unet)
                    logging.warning(f'Using {torch.cuda.device_count()} GPUs for testing Second stage {roi}')
               model_path = os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP2_FEATURE_ROOT) +'_second.pt')
               Unet = Unet.to(device)
               test_dataset = Test_Dataset(testing_paths, BENCHMARK_IMAGE, OUTPUT_DIR, roi, im_size, stage)
               test_loader = DataLoaderX(test_dataset, batch_size=1, shuffle=False)
               test_model(test_loader, Unet, model_path, roi, stage, adaptive_threshold = THRESHOLD)

     if BINARY_MAPS:
          build_binary_maps(rois, OUTPUT_DIR, LABELMAP_DIR, BENCHMARK_DICOM, os.path.join(BENCHMARK_IMAGE,'Images'), file_list = testing_paths, adaptive_threshold = THRESHOLD, first = True, second = True)
          print('Zipping...')
          shutil.make_archive(LABELMAP_DIR, 'zip', LABELMAP_DIR)
          print('Zipped')

def dicom2image(benchmark_dicom, benchmark_image, flip = False):
     if not os.path.exists(benchmark_image):
        os.makedirs(benchmark_image)
     if not os.path.exists(os.path.join(benchmark_image,'Images')):
        os.makedirs(os.path.join(benchmark_image,'Images'))
     file_list = os.listdir(benchmark_dicom)
     for name in tqdm(file_list):
          sub = os.path.join(benchmark_dicom,name)
          try:
               images = read_images(sub)
          except:
               print('error',sub)
               continue
          resized_images = resize_images(images)
          if flip:
               flip_images = np.flip(images,2)
               resized_flip_images = resize_images_labels(flip_images)

          hdf5_file = h5py.File(os.path.join(benchmark_image, 'Images', name + '.hdf5'), 'w')
          hdf5_file.create_dataset('images', data=images)
          hdf5_file.create_dataset('resized_images', data=resized_images)
          hdf5_file.close()
          if flip:
               hdf5_file = h5py.File(os.path.join(benchmark_image, 'Images', name + 'flip.hdf5'), 'w')
               hdf5_file.create_dataset('images', data=flip_images)
               hdf5_file.create_dataset('resized_images', data=resized_flip_images)
               hdf5_file.close()


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('--preprocess', type=bool, default=True)
     parser.add_argument('--first', type=bool, default=True)
     parser.add_argument('--second', type=bool, default=True)
     parser.add_argument('--labelmap', type=bool, default=True)
     args = parser.parse_args()

     rois = ['Bone_Mandiblez', 'Thyroidz', 'Brainstemz', 'Cavity_Oralz', 'Esophagusz', 'Eye_Lz',\
                    'Eye_Rz', 'Glnd_Submand_Lz', 'Glnd_Submand_Rz', 'Larynxz',  'LN_L_Ibz', 'LN_L_II-IVz', \
                    'LN_L_Vz', 'LN_R_Ibz', 'LN_R_II-IVz', 'LN_R_Vz', 'Lobe_Temporal_Lz', 'Lobe_Temporal_Rz', 'Musc_Constrictz',\
                     'Parotid_Lz', 'Parotid_Rz', 'SpinalCordz', 'Tracheaz']

     if args.preprocess:
          dicom2image(BENCHMARK_DICOM, BENCHMARK_IMAGE)

     main(args.first, args.second, args.labelmap, rois)