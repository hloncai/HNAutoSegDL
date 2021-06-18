import os, glob, time, h5py, torch
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize, warp, AffineTransform
from skimage.draw import polygon
from skimage import measure
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from .HN_Constants import *
from .metrics import *
from scipy.spatial.distance import directed_hausdorff

def normalize(im_input):
    im_output = im_input + 1000 # We want to have air value to be 0 since HU of air is -1000
    # Intensity crop
    im_output[im_output < 0] = 0
    im_output[im_output > 2000] = 2000 # Kind of arbitrary to select the range from -1000 to 1000 in HU
    im_output = im_output / 2000.0
    return im_output

def standardlize(im_input):
    mean = im_input.mean()
    std = im_input.std()
    im_output = (im_input-mean)/std
    return im_output

def resize_images_labels(images, labels):
    resized_images = resize_images(images)
    resized_labels = np.zeros((N_CLASSES, ALL_IM_SIZE[0], ALL_IM_SIZE[1], ALL_IM_SIZE[2]))
    size = (ALL_IM_SIZE[0], ALL_IM_SIZE[1] + CROP * 2, ALL_IM_SIZE[2] + CROP * 2)
    for z in range(N_CLASSES):
        resized_label = np.zeros(size, dtype=np.float32)
        roi = resize(labels[z,:,:,:], size, mode='constant')
        resized_label[roi >= 0.5] = 1
        resized_labels[z,:,:,:] = resized_label[:, CROP:-CROP, CROP:-CROP]
    return resized_images, resized_labels

def resize_images(images):
    size = (ALL_IM_SIZE[0], ALL_IM_SIZE[1] + CROP * 2, ALL_IM_SIZE[2] + CROP * 2)
    resized_images = resize(images, size, mode='constant')
    resized_images = resized_images[:, CROP:-CROP, CROP:-CROP]
    return resized_images

def get_tform_coords(im_size):
    coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
    coords = np.array([coords0 - im_size[0] / 2, coords1 - im_size[1] / 2, coords2 - im_size[2] / 2])
    return np.append(coords.reshape(3, -1), np.ones((1, np.prod(im_size))), axis=0)

def clean_contour(in_contour, is_prob=False):
    if is_prob:
        pred = (in_contour >= 0.5).astype(np.float32)
    else:
        pred = in_contour
    labels = measure.label(pred)
    area = []
    for l in range(1, np.amax(labels) + 1):
        area.append(np.sum(labels == l))
    arg_max_volume = np.argmax(area)
    max_volume = area[arg_max_volume]
    area[arg_max_volume] = 0
    arg_second_max_volume = np.argmax(area)
    second_max_volume = area[arg_second_max_volume]
    out_contour = in_contour
    #keep the largest and the second largest(more than half of largest) label group
    if second_max_volume < 0.5*max_volume:
        out_contour[np.logical_and(labels > 0, labels != arg_max_volume + 1)] = 0
    else:
        temp = np.logical_and(labels != arg_max_volume + 1, labels != arg_second_max_volume + 1)
        out_contour[np.logical_and(labels > 0, temp)] = 0
    return out_contour


def restore_labels(labels, roi, read_info, stage):
    if stage == 'first': 
        # added a temporal fix for Thyroid threshold calc
        try:
            labels = clean_contour(labels, is_prob=True)
        except: 
            labels = labels
        labels = np.pad(labels, ((0, 0), (CROP, CROP), (CROP, CROP)), 'constant')
        restored_labels = np.zeros(read_info['shape'], dtype=np.float32)
        order = ROI_ORDER.index(roi)
        size = (read_info['shape'][0].item(),read_info['shape'][1].item(),read_info['shape'][2].item())
        label = resize(labels.astype(np.float32), size, mode='constant')
        label[label >= 0.5] = 1
        label[label < 0.5] = 0
        restored_labels[label == 1] = order + 1
    else:
        labels = clean_contour(labels, is_prob=True)
        # Resize to extracted shape, then pad to original shape
        size = (read_info['extract_shape'][0].item(),read_info['extract_shape'][1].item(),read_info['extract_shape'][2].item())
        labels = resize(labels, size, mode='constant')
        restored_labels = np.zeros(read_info['shape'], dtype=np.float32)
        extract = read_info['extract']
        restored_labels[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]] = labels
    return restored_labels

def read_training_inputs(file, file_address, roi, im_size, stage, augmentation = False):

    image_file = os.path.join(file_address,'Images' , os.path.basename(file))
    image_h5 = h5py.File(image_file, 'r')
    file = os.path.join(file_address ,roi, os.path.basename(file))
    if stage == 'first':
        f_h5 = h5py.File(file, 'r')
        image = np.asarray(image_h5['resized_images'], dtype=np.float32)
        label = np.asarray(f_h5['resized_labels'], dtype=np.float32)
    else:
        f_h5 = h5py.File(file, 'r')
        image = np.asarray(image_h5['images'], dtype=np.float32)
        label = np.asarray(f_h5['labels'], dtype=np.float32)
    f_h5.close()
    image_h5.close()
    
    if stage == 'first':
        # Select all
        assert im_size == image.shape
        if augmentation:
            translation = [0, np.random.uniform(-8, 8), np.random.uniform(-8, 8)]
            rotation = euler2mat(np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
            scale = [1, np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]
            warp_mat = compose(translation, rotation, scale)
            tform_coords = get_tform_coords(im_size)
            w = np.dot(warp_mat, tform_coords)
            w[0] = w[0] + im_size[0] / 2
            w[1] = w[1] + im_size[1] / 2
            w[2] = w[2] + im_size[2] / 2
            warp_coords = w[0:3].reshape(3, im_size[0], im_size[1], im_size[2])
            final_image = warp(image, warp_coords)

            final_label = np.empty(im_size, dtype=np.float32)
            temp = warp(label.astype(np.float32), warp_coords)
            temp[temp < 0.5] = 0
            temp[temp >= 0.5] = 1
            final_label = temp
        else:
            final_image = image
            final_label = label

    else:
        # Select the roi
        roi_label = label.astype(np.float32)

        # Rotate the images and labels
        if augmentation:
            rotation = np.random.uniform(-15, 15)
            shear = np.random.uniform(-5, 5)
            tf = AffineTransform(rotation=np.deg2rad(rotation), shear=np.deg2rad(shear))
            for z in range(image.shape[0]):
                image[z] = warp(image[z], tf.inverse)
                roi_label[z] = warp(roi_label[z], tf.inverse)

        nz = np.nonzero(roi_label)
        extract = []
        for c in range(3):
            try:
                start = np.amin(nz[c])
            except:
                print(file)
                break
            end = np.amax(nz[c])
            r = end - start
            extract.append((np.maximum(int(np.rint(start - r * np.random.uniform(0.15, 0.25))), 0),
                            np.minimum(int(np.rint(end + r * np.random.uniform(0.15, 0.25))), image.shape[c])))
        extract_image = image[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]]
        extract_label = roi_label[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], 
                                    extract[2][0] : extract[2][1]]

        final_image = resize(extract_image, im_size, mode='constant')

        final_label = np.zeros(im_size, dtype=np.float32)
        lab = resize(extract_label, im_size, mode='constant')
        final_label[lab >= 0.5] = 1
    
    return final_image, final_label

def read_testing_inputs(file, image_path, output_path, roi, im_size, stage, extend = 0.2):

    if '.hdf5' not in file:
        file += '.hdf5'
    file = os.path.basename(file)
    image_file = os.path.join(image_path, 'Images', file)
    image_h5 = h5py.File(image_file, 'r')

    if stage == 'first':
        images = np.asarray(image_h5['resized_images'], dtype=np.float32)
        read_info = {}
        read_info['shape'] = np.asarray(image_h5['images'], dtype=np.float32).shape

    elif stage == 'second':
        images = np.asarray(image_h5['images'], dtype=np.float32)
        output = h5py.File(os.path.join(output_path, roi, roi+'_first_' + os.path.basename(file)),'r')
        predictions = np.asarray(output['predictions'], dtype=np.float32)
        output.close()
        # Select the roi
        roi_labels = predictions.astype(np.float32)
        nz = np.nonzero(roi_labels)
        extract = []
        for c in range(3):
            start = np.amin(nz[c])
            end = np.amax(nz[c])
            r = end - start
            extract.append((np.maximum(int(np.rint(start - r * extend)), 0),
                            np.minimum(int(np.rint(end + r * extend)), images.shape[c])))

        extract_images = images[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]]
        read_info = {}
        read_info['shape'] = images.shape
        read_info['extract_shape'] = extract_images.shape
        read_info['extract'] = extract
        images = resize(extract_images, im_size, mode='constant')
    
    image_h5.close()
    return images, read_info

def dice_numpy(probs,labels):
    eps = 1e-5
    dice_loss = 0
    intersection_prob = np.sum(probs* labels)
    union = np.sum(probs)+ np.sum(labels)
    dice_loss = 2*intersection_prob/union
    return dice_loss

def dice(probs,labels):
    eps = 1e-5
    dice_loss = 0
    slice_prob = probs.squeeze(1)
    slice_label = labels.squeeze(1)
    intersection_prob = torch.Tensor.sum(torch.Tensor.mul(slice_prob, slice_label), (1, 2, 3))
    union = eps + torch.Tensor.sum(slice_prob, (1, 2, 3)) + torch.Tensor.sum(slice_label, (1, 2, 3))
    dice_loss = 1 - 2 * torch.Tensor.mean(torch.Tensor.div(intersection_prob, union))
    return dice_loss
    
def read_spacing(dicom_path, name):
    if '.hdf5' in name:
        name = name.replace('.hdf5','')
    sub = os.path.join(dicom_path, name)
    dcms = glob.glob(os.path.join(sub, '*.dcm'))
    slice_file = [dcm for dcm in dcms if 'CT' in dcm]
    slices = [dicom.read_file(dcm) for dcm in slice_file]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    spacing = [float(x) for x in slices[0].PixelSpacing]
    thick = float(slices[0].SliceThickness)
    spacing.insert(2, thick)
    spacing = tuple(spacing)
    return spacing

def build_binary_maps(rois, output_path, labelmap_path, dicom_path=None, image_path=None, file_list = None, adaptive_threshold = False, first = True, second = True):
    # data path for RadComp testing data
    if image_path == None:
        image_path = 'Data/Images'
    if dicom_path == None:
        dicom_path = '/media/lingshu/data/AutoSegTrainingData_HN/DICOM/'
    if not os.path.exists(os.path.join(labelmap_path, 'Second_Labelmap')):
        os.makedirs(os.path.join(labelmap_path, 'Second_Labelmap'))
    if not os.path.exists(os.path.join(labelmap_path, 'First_Labelmap')):
        os.makedirs(os.path.join(labelmap_path, 'First_Labelmap'))

    if file_list == None:
        file_list = ['HN' + str(file_idx).zfill(3) for file_idx in range(1,20)]

    for name in tqdm(file_list):
        if '.hdf5' in name:
            name = name.replace('.hdf5','')
        if not os.path.exists(os.path.join(labelmap_path, 'Second_Labelmap', name)):
            os.makedirs(os.path.join(labelmap_path, 'Second_Labelmap', name))
        if not os.path.exists(os.path.join(labelmap_path, 'First_Labelmap', name)):
            os.makedirs(os.path.join(labelmap_path, 'First_Labelmap', name))
        image_h5 = h5py.File(os.path.join(image_path, name + '.hdf5'), 'r')
        images = np.asarray(image_h5['images'], dtype=np.float32)
        reader = sitk.ImageSeriesReader()
        for subdir, _, _ in os.walk(os.path.join(dicom_path, name)):
            orig_images = sitk.ReadImage(reader.GetGDCMSeriesFileNames(subdir))
        if first:
            for roi in rois:
                first_labels = np.zeros_like(images, dtype=np.float32)
                f = h5py.File(os.path.join(output_path, roi, roi + '_first_' + name + '.hdf5'), 'r')
                predictions = np.asarray(f['predictions'], dtype=np.float32)
                f.close()
                first_labels[predictions != 0] = 1
                labels = restore_spacing(dicom_path, name, first_labels, roi)
                img = sitk.GetImageFromArray(labels)
                img.CopyInformation(orig_images)
                sitk.WriteImage(img, os.path.join(labelmap_path,'First_Labelmap', name, name +'_first_' + roi + '_labels.mha'))
        if second:
            for roi in rois:
                labels = np.zeros_like(images, dtype=np.float32)
                f = h5py.File(os.path.join(output_path, roi, roi + '_second_' + name + '.hdf5'), 'r')
                probs = np.asarray(f['probs'], dtype=np.float32)
                f.close()
                threshold = THRESHOLD[roi] if adaptive_threshold else 0.5
                labels[probs < threshold] = 0 # Ignore those classfied as background
                labels[probs >= threshold] = 1
                labels = restore_spacing(dicom_path, name, labels, roi)
                img = sitk.GetImageFromArray(labels)
                img.CopyInformation(orig_images)
                sitk.WriteImage(img, os.path.join(labelmap_path, 'Second_Labelmap', name, name +'_second_' + roi + '_labels.mha'))

def get_dimension(dicom_path, name):
    for subdir, dirs, files in os.walk(os.path.join(dicom_path, name)):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        slice_file = [dcm for dcm in dcms if 'CT' in dcm]
        slices = [dicom.read_file(dcm) for dcm in slice_file]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        images = images + slices[0].RescaleIntercept
    return images.shape[0]

def restore_spacing(dicom_path, name, label, roi):

    z_size = get_dimension(dicom_path, name)
    spacing = read_spacing(dicom_path, name)
    PIXEL_SPACING = 0.9765625
    INPLANE_SIZE = 512
    SLICE_THICKNESS = 2

    inplane_scale = PIXEL_SPACING / spacing[0]
    inplane_size = int(np.rint(inplane_scale * INPLANE_SIZE / 2) * 2)
    z_scale = SLICE_THICKNESS / spacing[2]

    if inplane_size != INPLANE_SIZE or z_scale != 1:
        label = resize(label, (z_size, inplane_size, inplane_size), mode='constant')
        restored_labels = np.zeros_like(label, dtype=np.int8)
        restored_labels[label >= 0.5] = ROI_ORDER.index(roi) + 1
        if inplane_size != INPLANE_SIZE:
            if inplane_size > INPLANE_SIZE:
                crop = int((inplane_size - INPLANE_SIZE) / 2)
                restored_labels = restored_labels[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            else:
                pad = int((INPLANE_SIZE - inplane_size) / 2)
                restored_labels = np.pad(restored_labels, ((0, 0), (pad, pad), (pad, pad)))
    else:
        restored_labels = np.zeros_like(label, dtype=np.int8)
        restored_labels[label >= 0.5] = ROI_ORDER.index(roi) + 1
    return restored_labels

def compute_hausdorff(mask_gt,mask_pred,spacing_mm):
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
    h_distance = compute_robust_hausdorff(surface_distances, 95)
    return h_distance