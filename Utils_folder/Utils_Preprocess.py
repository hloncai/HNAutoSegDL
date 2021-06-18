import os, glob, h5py, random, json
import numpy as np
import pydicom as dicom
from skimage.draw import polygon
from skimage.transform import resize
from .HN_Constants import *
from .Utils import *
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def weight_size(file_name, roi):
    file_name = os.path.basename(file_name)
    f_h5 = h5py.File(os.path.join('Data',roi,file_name), 'r')
    labels = np.asarray(f_h5['labels'], dtype=np.float32)
    resized_labels = np.asarray(f_h5['resized_labels'], dtype=np.float32)
    f_h5.close()

    resize_roi_count = np.sum(resized_labels==1)
    resize_bg_count = resized_labels.shape[0] * resized_labels.shape[1] * resized_labels.shape[2] - resize_roi_count

    ROI_size = np.zeros(3, dtype = np.int16)
    nz = np.nonzero(labels)
    extract = []
    for c in range(3):
        start = np.amin(nz[c])
        end = np.amax(nz[c])
        r = np.abs(end - start)
        ROI_size[c] = int(r)
        extract.append((np.maximum(int(np.rint(start - r * 0.2)), 0),
                            np.minimum(int(np.rint(end + r * 0.2)), labels.shape[c])))

    extract_labels = labels[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], 
                                extract[2][0] : extract[2][1]]

    extract_labels_numpy = np.array(extract_labels)
    roi_count = np.sum(extract_labels_numpy==1)
    
    bg_count = extract_labels_numpy.shape[0] * extract_labels_numpy.shape[1] * extract_labels_numpy.shape[2] - roi_count
    
    return roi_count, bg_count, ROI_size, resize_roi_count, resize_bg_count

def size_ratio(train_file_lists, ROI_ORDER):
    ratios = []
    bg_ratios = []
    average_ROI_sizes = []
    pd_ROI_sizes = pd.DataFrame(columns = ROI_ORDER)
    none_space = []
    for _ in range(1000):
        none_space.append(None)
    pd_ROI_sizes[ROI_ORDER[0]] = none_space

    pd_roi_count = pd.DataFrame(columns = ROI_ORDER)
    none_space = []
    for _ in range(1000):
        none_space.append(None)
    pd_roi_count[ROI_ORDER[0]] = none_space

    with tqdm(ROI_ORDER) as t:
        for roi in t:
            training_paths = [path for path in train_file_lists[roi] if type(path) is type('')]
            if len(training_paths) == 0:
                continue
            ratio = []
            bg_ratio = []
            roi_counts = []
            ROI_sizes = []
            for path in training_paths:
                if '.hdf5' not in path:
                    path += '.hdf5'
                roi_count, bg_count, ROI_size, resize_roi_count, resize_bg_count = weight_size(path, roi)
                try:
                    ratio.append(bg_count/roi_count)
                    bg_ratio.append(resize_bg_count/resize_roi_count)
                except:
                    print(roi,'path',path)
                roi_counts.append([path, roi_count])
                ROI_sizes.append(ROI_size)
                t.set_postfix(roi=roi, path=path, ROI_size = ROI_size, roi_count = roi_count)
            ratios.append(np.round(np.mean(ratio),2))
            bg_ratios.append(np.round(np.mean(bg_ratio),2))
            average_ROI_size = np.round(np.mean(ROI_sizes,0)).astype(np.int16)
            ROI_sizes.append(average_ROI_size)
            average_ROI_sizes.append(average_ROI_size.tolist())
            pd_roi_count.update(pd.DataFrame({roi: roi_counts}))
            pd_ROI_sizes.update(pd.DataFrame({roi: ROI_sizes}))
    pd_ROI_sizes.to_csv('./Filelist/ROI_Dimension.csv',index=False)
    pd_roi_count.to_csv('./Filelist/ROI_Pixel_Number.csv',index=False)
    with open('./Utils_folder/HN_Constants.json', 'w') as f:
        json.dump([ratios,bg_ratios,average_ROI_sizes],f)

def train_filelist(DIR,ROI_ORDER):
    file_list = pd.read_csv('./Filelist/ALL_Filelist.csv')
    train_list, test_list = train_test_split(file_list, test_size=0.3, shuffle=True)
    train_list.sort()
    test_list.sort()
    train_pd = pd.DataFrame(train_list)
    test_pd = pd.DataFrame(test_list)
    train_pd.to_csv('./Filelist/Train_Filelist.csv',index=False)
    test_pd.to_csv('./Filelist/Test_Filelist.csv',index=False)

    data_paths = [path for path in train_pd['0'] if type(path) is type('')]

    ROI_ORDER = [str(x) for x in ROI_ORDER]
    train_pd = pd.DataFrame(columns = ROI_ORDER)
    val_pd = pd.DataFrame(columns = ROI_ORDER)
    none_space = []
    for _ in range(len(data_paths)):
        none_space.append(None)
    train_pd[ROI_ORDER[0]] = none_space
    val_pd[ROI_ORDER[0]] = none_space

    for roi in tqdm(ROI_ORDER):
        file_list = []
        for path in data_paths:
            path = path + '.hdf5'
            path = os.path.join(DIR, roi, path)
            try:
                f_h5 = h5py.File(path, 'r')
                label = np.asarray(f_h5['resized_labels'], dtype=np.float32)
                if np.sum(label)!=0:
                    file_list.append(os.path.basename(path))
                f_h5.close()
            except:
                continue
        
        if len(file_list) <= 2:
            continue
        train_list, val_list = train_test_split(file_list, test_size=0.2, shuffle=True)
        train_list.sort()
        val_list.sort()
        train_pd.update(pd.DataFrame({roi: train_list}))
        val_pd.update(pd.DataFrame({roi: val_list}))
    train_pd.to_csv('./Filelist/Train_Filelist_ROI.csv',index=False)
    val_pd.to_csv('./Filelist/Val_Filelist_ROI.csv',index=False)

def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        temp = structure.StructureSetROISequence[i].ROIName
        if temp in ROI_ORDER:
            contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
            contour['name'] = structure.StructureSetROISequence[i].ROIName
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            try:
                contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            except:
                contour['contours'] = None
            contours.append(contour)
        else:
            continue
    return contours

def get_labels(contours, shape, slices):
    z = [np.around(s.ImagePositionPatient[2], 1) for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]   
    label_map = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name'])
        if con['contours'] != None:
            for c in con['contours']:
                nodes = np.array(c).reshape((-1, 3))
                # assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                z_index = z.index(np.around(nodes[0, 2], 1))
                r = (nodes[:, 1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r, c)
                label_map[num, z_index, rr, cc] = 1
        else:
            # print(con['name'],'not available')
            label_map[num, :, :, :] = 0
    return label_map

def read_images_labels(path):
    # Read the images and labels from a folder containing both dicom files
    for subdir, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        structure_file = [dcm for dcm in dcms if 'RS' in dcm]
        structure = dicom.read_file(structure_file[0])
        contours = read_structure(structure)
        slice_file = [dcm for dcm in dcms if 'CT' in dcm]
        slices = [dicom.read_file(dcm) for dcm in slice_file]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        images = images + slices[0].RescaleIntercept
    
    images = normalize(images)
    shape = images.shape
    shape = (N_CLASSES,) + shape
    labels = get_labels(contours, shape, slices)

    inplane_scale = slices[0].PixelSpacing[0] / PIXEL_SPACING
    inplane_size = int(np.rint(inplane_scale * slices[0].Rows / 2) * 2)
    z_scale = slices[0].SliceThickness / SLICE_THICKNESS
    z_size = int(np.rint(z_scale * images.shape[0]))
    
    if inplane_size != INPLANE_SIZE or z_scale != 1:
        images = resize(images, (z_size, inplane_size, inplane_size), mode='constant')
        new_labels = np.zeros((N_CLASSES, z_size, inplane_size, inplane_size), dtype=np.float32)
        for index in range(N_CLASSES):
            roi_labels = np.zeros_like(images, dtype=np.float32)
            temp = labels[index,:,:,:]
            roi = resize(temp.astype(np.float32), (z_size, inplane_size, inplane_size), mode='constant')
            roi_labels[roi >= 0.5] = 1
            new_labels[index,:,:,:] = roi_labels
        # labels = new_labels
        if inplane_size != INPLANE_SIZE:
            if inplane_size > INPLANE_SIZE:
                crop = int((inplane_size - INPLANE_SIZE) / 2)
                images = images[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
                labels = new_labels[:, :, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            else:
                pad = int((INPLANE_SIZE - inplane_size) / 2)
                images = np.pad(images, ((0, 0), (pad, pad), (pad, pad)))
                labels = np.pad(new_labels, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            labels = new_labels
        
    return images, labels

def read_images(path):
    # Read the images and labels from a folder containing both dicom files
    for subdir, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        slice_file = [dcm for dcm in dcms if 'CT' in dcm]
        slices = [dicom.read_file(dcm) for dcm in slice_file]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        images = images + slices[0].RescaleIntercept
    
    images = normalize(images)

    inplane_scale = slices[0].PixelSpacing[0] / PIXEL_SPACING
    inplane_size = int(np.rint(inplane_scale * slices[0].Rows / 2) * 2)
    z_scale = slices[0].SliceThickness / SLICE_THICKNESS
    z_size = int(np.rint(z_scale * images.shape[0]))
    
    if inplane_size != INPLANE_SIZE or z_scale != 1:
        images = resize(images, (z_size, inplane_size, inplane_size), mode='constant')
        if inplane_size != INPLANE_SIZE:
            if inplane_size > INPLANE_SIZE:
                crop = int((inplane_size - INPLANE_SIZE) / 2)
                images = images[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            else:
                pad = int((INPLANE_SIZE - inplane_size) / 2)
                images = np.pad(images, ((0, 0), (pad, pad), (pad, pad)))
        
    return images

def preprocess_list(input_path, file_list, output_path, flip):
    for name in tqdm(file_list):
        sub = os.path.join(input_path,name)
        #images, labels = read_images_labels(sub)
        try:
            images, labels = read_images_labels(sub)
        except:
            print('error',sub)
            continue
        resized_images, resized_labels = resize_images_labels(images, labels)
        if flip:
            flip_images = np.flip(images,2)
            flip_labels = np.flip(labels,3)
            resized_flip_images, resized_flip_labels = resize_images_labels(flip_images, flip_labels)

        hdf5_file = h5py.File(os.path.join(output_path, 'Images', name + '.hdf5'), 'w')
        hdf5_file.create_dataset('images', data=images)
        hdf5_file.create_dataset('resized_images', data=resized_images)
        hdf5_file.close()
        if flip:
            hdf5_file = h5py.File(os.path.join(output_path, 'Images', name + 'flip.hdf5'), 'w')
            hdf5_file.create_dataset('images', data=flip_images)
            hdf5_file.create_dataset('resized_images', data=resized_flip_images)
            hdf5_file.close()
        for roi in ROI_ORDER:
            if not os.path.exists(os.path.join(output_path,roi)):
                os.makedirs(os.path.join(output_path,roi))
            index = ROI_ORDER.index(roi)
            slice_resized_label = resized_labels[index,:,:,:]
            slice_label = labels[index,:,:,:]
            hdf5_file = h5py.File(os.path.join(output_path,roi, (name + '.hdf5')), 'w')
            hdf5_file.create_dataset('labels', data=slice_label, compression="gzip", compression_opts=5)
            hdf5_file.create_dataset('resized_labels', data=slice_resized_label, compression="gzip", compression_opts=5)
            hdf5_file.close()
            if flip:
                flip_index = FLIP_ROI_ORDER.index(roi)
                flip_slice_resized_label = resized_flip_labels[flip_index,:,:,:]
                flip_slice_label = flip_labels[flip_index,:,:,:]
                hdf5_file = h5py.File(os.path.join(output_path,roi, (name + 'flip.hdf5')), 'w')
                hdf5_file.create_dataset('labels', data=flip_slice_label, compression="gzip", compression_opts=5)
                hdf5_file.create_dataset('resized_labels', data=flip_slice_resized_label, compression="gzip", compression_opts=5)
                hdf5_file.close()

def preprocess(ALL_IM_SIZE, input_path, flip, data_path = 'Data'):
    file_list = os.listdir(input_path)
    file_list.sort()
    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv('./Filelist/ALL_Filelist.csv',index=False)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(os.path.join(data_path,'Images')):
        os.makedirs(os.path.join(data_path,'Images'))

    print('Generate HDF5 file')
    preprocess_list(input_path, file_list, data_path, flip)