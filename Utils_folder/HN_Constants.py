import json
DICOM_DIR = '/media/lingshu/data/AutoSegTrainingData_HN/DICOM/'
ALL_IM_SIZE = (112, 208, 208)
CROP = 24
INPLANE_SIZE = 512
PIXEL_SPACING = 0.9765625
SLICE_THICKNESS = 2
STEP1_FEATURE_ROOT = 24
STEP2_FEATURE_ROOT = 24
DROPOUT_RATIO = 0.5
FISRT_BATCHSIZE = 2
SECOND_BATCHSIZE = 30
ROI_ORDER = ['Bone_Mandiblez', 'Thyroidz', 'Brainstemz', 'Cavity_Oralz', 'Esophagusz', 'Eye_Lz',\
                    'Eye_Rz', 'Glnd_Submand_Lz', 'Glnd_Submand_Rz', 'Larynxz',  'LN_L_Ibz', 'LN_L_II-IVz', 'LN_L_RPz', \
                    'LN_L_Vz', 'LN_R_Ibz', 'LN_R_II-IVz', 'LN_R_RPz', 'LN_R_Vz', 'Lobe_Temporal_Lz', 'Lobe_Temporal_Rz', 'Musc_Constrictz',\
                     'Parotid_Lz', 'Parotid_Rz', 'SpinalCordz', 'Tracheaz']
FLIP_ROI_ORDER = ['Bone_Mandiblez', 'Thyroidz', 'Brainstemz', 'Cavity_Oralz', 'Esophagusz',  'Eye_Rz',\
                    'Eye_Lz', 'Glnd_Submand_Rz', 'Glnd_Submand_Lz', 'Larynxz',  'LN_R_Ibz', 'LN_R_II-IVz', 'LN_R_RPz', \
                    'LN_R_Vz', 'LN_L_Ibz', 'LN_L_II-IVz', 'LN_L_RPz', 'LN_L_Vz', 'Lobe_Temporal_Rz', 'Lobe_Temporal_Lz', 'Musc_Constrictz',\
                     'Parotid_Rz', 'Parotid_Lz', 'SpinalCordz', 'Tracheaz']
FIRST_THRESHOLD = {'Bone_Mandiblez':0.5,'Thyroidz':0.5,'Parotid_Rz':0.7,'Parotid_Lz':0.65, 'LN_L_II-IVz':0.55, 'SpinalCordz':0.45, 'Esophagusz':0.65, 'Lobe_Temporal_Rz':0.75, 'Musc_Constrictz':0.55, 'Brainstemz':0.75, 'Cavity_Oralz':0.55, 'Glnd_Submand_Lz':0.95, 'LN_R_II-IVz':0.35, 'LN_R_Ibz':0.8, 'LN_L_Ibz':0.55, 'Glnd_Submand_Rz':0.95,  'Lobe_Temporal_Lz':0.85, 'LN_L_Vz':0.75, 'LN_R_Vz':0.85, 'Eye_Lz':0.7, 'Eye_Rz':0.95, 'Tracheaz':0.95, 'Larynxz':0.95}
SECOND_THRESHOLD = {'Bone_Mandiblez':0.5,'Thyroidz':0.5,'Parotid_Rz':0.7,'Parotid_Lz':0.1, 'LN_L_II-IVz':0.75, 'SpinalCordz':0.45, 'Esophagusz':0.65, 'Lobe_Temporal_Rz':0.7, 'Musc_Constrictz':0.55, 'Brainstemz':0.75, 'Cavity_Oralz':0.8, 'Glnd_Submand_Lz':0.85, 'LN_R_II-IVz':0.6, 'LN_R_Ibz':0.85, 'LN_L_Ibz':0.8, 'Glnd_Submand_Rz':0.8,  'Lobe_Temporal_Lz':0.85, 'LN_L_Vz':0.85, 'LN_R_Vz':0.85, 'Eye_Lz':0.7, 'Eye_Rz':0.1, 'Tracheaz':0.7, 'Larynxz':0.75}
N_CLASSES = len(ROI_ORDER)
with open('Utils_folder/HN_Constants.json', 'r') as f:
    WEIGHTS, BG_WEIGHTS, IM_SIZES = json.load(f)