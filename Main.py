import os, sys, torch, logging, shutil, h5py, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Utils_folder.Utils import *
from Utils_folder.HN_Constants import *
from Model.Model_First import AttU_Net_3layer
from Model.Model_Second import AttU_Net_4layer
from Model.Model_Old import UNet3D
from Utils_folder.Dataset_Util import Train_Dataset, Test_Dataset
from tqdm import tqdm
import pandas as pd
from Utils_folder.Augmentation import AugmentationPipeline
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

LOAD_OLD_MODEL = True
THRESHOLD = False
MORE_STEP = 2000
DATA_DIR = 'Data'
OUTPUT_DIR = 'Data_Output'
LABELMAP_DIR = 'Data_Labelmaps'
LOSS_TYPE = 'mixed'  #support three type of loss/ dice / crossentropy / mixed /
AUGMENTATION = False
CHECKPOINT_DIR = 'Checkpoint'
OLD_CHECKPOINT_DIR = 'checkpoint_3layer'
METRIC_DIR = 'Metric'
LOG_DIR = 'Logs/' + str(STEP2_FEATURE_ROOT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEP = 8000
if LOAD_OLD_MODEL:
    STEP1_FEATURE_ROOT = 48
    DROPOUT_RATIO = 0.3

MyAug = AugmentationPipeline()

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def train_model(train_dataloader, val_dataloader, optimizer, scheduler, net, BCE_Loss, roi, stage, path, validation, load_model, logger_address, max_step, loss_type = 'dice', augmentation = False):
    step = 0
    val_step = 0
    if load_model:
        checkpoint = torch.load(path)#load new and full type model
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step'] + 1
        val_step = checkpoint['val_step'] + 1
        max_step = step + MORE_STEP

    writer = SummaryWriter(LOG_DIR)#enabel logger for tensorboard

    with tqdm(range(1000)) as t:#due to unstable number of rois, the number of update for training is fixed instead of the number of epoch
        for epoch in t:
            if step > max_step:
                break
            for data in train_dataloader:
                net.train()
                image = data[0].to(device)
                label = data[1].to(device)
                if augmentation: #perform augmentation by pytorch tensor pipeline
                    image, label = MyAug(image,label)
                optimizer.zero_grad()
                output = net(image)
                dice_loss = dice(output, label)
                crossentropy_loss = BCE_Loss(output, label)
                if LOSS_TYPE == 'dice':
                    loss = dice_loss
                elif LOSS_TYPE == 'crossentropy':
                    loss = crossentropy_loss
                elif LOSS_TYPE == 'mixed':
                    loss = crossentropy_loss + dice_loss*2
                loss.backward()
                t.set_description("Step %i"%step)
                t.set_postfix(dice_loss=dice_loss.item(), crossentropy_loss=crossentropy_loss.item())#show loss in tqdm progess bar
                optimizer.step()
                writer.add_scalars(logger_address, {'dice': dice_loss.item(),
                                        'crossentropy':crossentropy_loss.item()}, step)
                step += 1
                scheduler.step()
                if step > max_step:
                    break

            if validation and epoch % 3 == 0:
                with torch.no_grad():
                    net.eval()
                    for data in val_dataloader:
                        image = data[0].to(device)
                        label = data[1].to(device)
                        output = net(image)
                        dice_loss = dice(output, label)
                        crossentropy_loss = BCE_Loss(output, label)
                        writer.add_scalars(logger_address+'val', {'dice': dice_loss.item(),
                                        'crossentropy':crossentropy_loss.item()}, val_step)
                        val_step += 1

            state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'step':step, 'val_step': val_step}
            if stage == 'first':
                torch.save(state, os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP1_FEATURE_ROOT) +'_first.pt'))
            elif stage == 'second':
                torch.save(state, os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP2_FEATURE_ROOT) +'_second.pt'))

def test_and_metric(file_name, output, read_info, roi, stage, adaptive_threshold):
    dice_losses = []
    distances = []
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    output_file = os.path.join(OUTPUT_DIR, roi, roi + '_'+ stage + '_' + os.path.basename(file_name) )
    f_h5 = h5py.File(output_file, 'w')
    output = output.cpu()
    output = output.squeeze()
    slice_prob = output
    probs = slice_prob.numpy()
    slice_prediction = np.zeros(probs.shape, dtype=np.int8)
    label_h5 = h5py.File(os.path.join(DATA_DIR, roi, file_name), 'r')
    if stage == 'first':
        label = np.asarray(label_h5['resized_labels'], dtype=np.float32)
        threshold = FIRST_THRESHOLD[roi] if adaptive_threshold else 0.5
        slice_prediction[probs < threshold] = 0
        slice_prediction[probs >= threshold] = 1
        prediction = restore_labels(slice_prediction, roi, read_info, stage)
    else:
        label = np.asarray(label_h5['labels'], dtype=np.float32)
        probs = restore_labels(probs, roi, read_info, stage)
        prediction = np.round(probs)
    label_h5.close()

    f_h5.create_dataset('probs', data=probs, compression="gzip", compression_opts=9)
    f_h5.create_dataset('predictions', data=prediction, compression="gzip", compression_opts=9)
    f_h5.close()
    spacing = read_spacing(DICOM_DIR, file_name)
    for threshold in np.arange(1,10,0.5):
        output = np.zeros(probs.shape, dtype=np.int8)
        threshold = threshold/10
        output[probs < threshold] = 0 # Ignore those classfied as background
        output[probs >= threshold] = 1
        dice_loss = dice_numpy(output,label)
        distance = compute_hausdorff(label, output, spacing)
        # t.set_postfix(threshold = threshold,dice=dice_loss,distance = distance)
        dice_losses.append(dice_loss)
        distances.append(distance)
    return dice_losses, distances


def test_model(test_dataloader, net, path, roi, stage, adaptive_threshold = False):
    ###Store performance metric###
    if not os.path.exists(os.path.join(METRIC_DIR,roi)):
        os.makedirs(os.path.join(METRIC_DIR,roi))
    if not os.path.exists(os.path.join(METRIC_DIR,roi,stage)):
        os.makedirs(os.path.join(METRIC_DIR,roi,stage))
    dice_losses_allfile = []
    distance_allfile = []
    ###...........................###
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
                dice_losses, distances = test_and_metric(file_name, output, read_info, roi, stage, adaptive_threshold)
                best_threshold = np.argmax(dice_losses)
                t.set_postfix(threshold = best_threshold, dice = dice_losses[best_threshold], distance = distances[best_threshold])
                if (np.sum(dice_losses) >= 0.01):
                    dice_losses_allfile.append(dice_losses)
                    distance_allfile.append(distances)
        dice_losses_allfile = np.array(dice_losses_allfile)
        file_lists = pd.DataFrame(dice_losses_allfile)
        file_lists.to_csv(os.path.join(METRIC_DIR, roi, stage, 'dice_test_'  + roi + '.csv'),index=False)
        
        distance_allfile = np.array(distance_allfile)
        distance_lists = pd.DataFrame(distance_allfile)
        distance_lists.to_csv(os.path.join(METRIC_DIR ,roi, stage, 'distance_test_'  + roi + '.csv'),index=False)

def main(FIRST_STAGE, SECOND_STAGE, TRAIN, BINARY_MAPS, VILDATION, LOAD_MODEL, rois):
    TEST = not TRAIN
    
    train_file_lists = pd.read_csv('Filelist/Train_Filelist_ROI.csv')
    test_file_lists = pd.read_csv('Filelist/Test_Filelist.csv')
    val_file_lists = pd.read_csv('Filelist/Val_Filelist_ROI.csv')

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(METRIC_DIR):
        os.makedirs(METRIC_DIR)
    
    testing_paths = [path for path in test_file_lists['0'] if type(path) is type('')]
    #testing_paths =['HN004.hdf5', 'HN023.hdf5']

    for roi in rois:
        index = ROI_ORDER.index(roi)
        if not os.path.exists(os.path.join(CHECKPOINT_DIR,roi)):
            os.makedirs(os.path.join(CHECKPOINT_DIR,roi))
        if not os.path.exists(os.path.join(OUTPUT_DIR, roi)):
            os.makedirs(os.path.join(OUTPUT_DIR, roi))
        if FIRST_STAGE:
            stage = 'first'
            if LOAD_OLD_MODEL:
                Unet_first = UNet3D(in_channels = 1, out_channels = 1, final_sigmoid = True, f_maps = 48, layer_order='crg', num_groups = 8, num_levels = 3, is_segmentation = True, testing = True)
            else:
                Unet_first = AttU_Net_3layer(feature = STEP1_FEATURE_ROOT, P = DROPOUT_RATIO, activate = True)
            if torch.cuda.device_count() > 1 and not device.type == 'cpu':
                Unet_first = nn.DataParallel(Unet_first)
            Unet_first = Unet_first.to(device)
            if TRAIN:
                logging.warning('Training on first layer '+ roi)
                training_paths = [path for path in train_file_lists[roi] if type(path) is type('')]
                val_paths = [path for path in val_file_lists[roi] if type(path) is type('')]

                if VILDATION:
                    dataset = Train_Dataset(DATA_DIR, training_paths, roi, ALL_IM_SIZE, stage)
                    train_loader = DataLoaderX(dataset, batch_size = FISRT_BATCHSIZE, shuffle = True, num_workers = 2)
                    val_dataset = Train_Dataset(DATA_DIR, val_paths, roi, ALL_IM_SIZE, stage)
                    val_loader = DataLoaderX(val_dataset, batch_size = 2, shuffle = True, num_workers = 2)
                else:
                    training_paths = training_paths + val_paths
                    dataset = Train_Dataset(DATA_DIR, training_paths, roi, ALL_IM_SIZE, stage)
                    train_loader = DataLoaderX(dataset, batch_size = FISRT_BATCHSIZE, shuffle = True, num_workers = 2)
                    val_loader = None

                BCE_Loss = nn.BCELoss(torch.Tensor((BG_WEIGHTS[index],)).to(device))

                if LOAD_MODEL:
                    optimizer = optim.Adam(Unet_first.parameters(), lr=0.001, weight_decay=0.001)
                    model_path = os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP1_FEATURE_ROOT) +'_first.pt')
                    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [2000, 6000], gamma=0.1)
                else:
                    optimizer = optim.Adam(Unet_first.parameters(), lr=0.001, weight_decay=0.001)
                    model_path = None
                    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [2000,], gamma=0.1)
                torch.cuda.empty_cache()
                train_model(train_loader, val_loader, optimizer, scheduler, Unet_first, BCE_Loss, roi, stage, model_path, VILDATION, LOAD_MODEL, logger_address = 'First_'+roi, max_step = MAX_STEP, loss_type = LOSS_TYPE, augmentation = AUGMENTATION)
                torch.cuda.empty_cache()
            if TEST:
                logging.warning('Testing on first layer '+ roi)
                model_path = os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP1_FEATURE_ROOT) +'_first.pt')
                test_dataset = Test_Dataset(testing_paths, DATA_DIR, OUTPUT_DIR, roi, ALL_IM_SIZE, stage)
                test_loader = DataLoaderX(test_dataset, batch_size=1, shuffle = False)
                test_model(test_loader, Unet_first, model_path, roi, stage, adaptive_threshold = False)

        im_size = tuple([int(x/8)*8 for x in IM_SIZES[index]])
        if SECOND_STAGE:
            stage = 'second'
            Unet = AttU_Net_4layer(feature = STEP2_FEATURE_ROOT, P = DROPOUT_RATIO, activate = True)
            if torch.cuda.device_count() > 1 and not device.type == 'cpu':
                Unet = nn.DataParallel(Unet)
            Unet = Unet.to(device)
            if TRAIN:
                logging.warning('Training on second layer '+ roi)
                training_paths = [path for path in train_file_lists[roi] if type(path) is type('')]
                val_paths = [path for path in val_file_lists[roi] if type(path) is type('')]
                if VILDATION:
                    dataset = Train_Dataset(DATA_DIR, training_paths, roi, im_size, stage)
                    train_loader = DataLoaderX(dataset, batch_size = SECOND_BATCHSIZE, shuffle = True, num_workers=10)
                    val_dataset = Train_Dataset(DATA_DIR, val_paths, roi, im_size, stage)
                    val_loader = DataLoaderX(val_dataset, batch_size= 2, shuffle = True, num_workers=2)
                else:
                    training_paths = training_paths + val_paths
                    dataset = Train_Dataset(DATA_DIR, training_paths, roi, im_size, stage)
                    train_loader = DataLoaderX(dataset, batch_size = SECOND_BATCHSIZE, shuffle = True, num_workers=10)
                    val_loader =None

                BCE_Loss = nn.BCELoss(torch.Tensor((WEIGHTS[index],)).to(device))

                if LOAD_MODEL:
                    optimizer = optim.Adam(Unet.parameters(), lr=0.0001,weight_decay=0.001)
                    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2000,], gamma=0.1)
                else:
                    optimizer = optim.Adam(Unet.parameters(), lr=0.001,weight_decay=0.001)
                    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2000,4000], gamma=0.1)
                model_path = os.path.join(CHECKPOINT_DIR, roi,  roi + '_'+ str(STEP2_FEATURE_ROOT) +'_second.pt')
                train_model(train_loader, val_loader, optimizer, scheduler, Unet, BCE_Loss, roi, stage, model_path, VILDATION,  LOAD_MODEL, logger_address =  'Second_'+roi, max_step = MAX_STEP, loss_type = LOSS_TYPE, augmentation = AUGMENTATION)
                torch.cuda.empty_cache()
            if TEST:
                logging.warning('Testing on second layer '+ roi)
                model_path = os.path.join(CHECKPOINT_DIR, roi, roi + '_'+ str(STEP2_FEATURE_ROOT) +'_second.pt')
                test_dataset = Test_Dataset(testing_paths, DATA_DIR, OUTPUT_DIR, roi, im_size, stage)
                test_loader = DataLoaderX(test_dataset, batch_size=1, shuffle=False)
                test_model(test_loader, Unet, model_path, roi, stage, adaptive_threshold = THRESHOLD)

    if BINARY_MAPS:
        build_binary_maps(rois, OUTPUT_DIR, LABELMAP_DIR, file_list = testing_paths, adaptive_threshold = THRESHOLD, first = True, second = True)
        print('Zipping...')
        shutil.make_archive(LABELMAP_DIR, 'zip', LABELMAP_DIR)
        print('Zipped')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=bool, default=True)
    parser.add_argument('--second', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--labelmap', type=bool, default=False)
    parser.add_argument('--vildation', type=bool, default=False)
    parser.add_argument('--continune', type=bool, default=True)
    args = parser.parse_args()

    rois = ['Brainstemz', 'Cavity_Oralz', 'Esophagusz', 'Eye_Lz',\
                    'Eye_Rz', 'Glnd_Submand_Lz', 'Glnd_Submand_Rz', 'Larynxz',  'LN_L_Ibz', 'LN_L_II-IVz',  \
                    'LN_L_Vz', 'LN_R_Ibz', 'LN_R_II-IVz',  'LN_R_Vz', 'Lobe_Temporal_Lz', 'Lobe_Temporal_Rz', 'Musc_Constrictz',\
                     'Parotid_Lz', 'Parotid_Rz', 'SpinalCordz', 'Tracheaz']
                     
    main(args.first, args.second, args.train, args.labelmap, args.vildation, args.continune, rois)