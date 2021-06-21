"""
This module contains functions for creating DICOM RT Structure Set files from binary mask 
segmentations of the CT volume. The files can be imported to Eclipse and will link correctly
with the existing patient's scans.

Author: Scott Ingram
"""

import os, numpy as np, SimpleITK as itk, pydicom as pd, cv2 as cv

UID_PREFIX = '1.2.826.0.1.3680043.10.308.'
CT_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.2'
RS_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.481.3'
OLD_NAMES = ['Bone_Mandiblez',
             'Brainstemz',
             'Cavity_Oralz',
             'Esophagusz',
             'Eye_Lz',
             'Eye_Rz',
             'Glnd_Submand_Lz',
             'Glnd_Submand_Rz',
             'Larynxz',
             'LN_L_Ibz',
             'LN_L_II-IVz',
             'LN_L_Vz',
             'LN_R_Ibz',
             'LN_R_II-IVz',
             'LN_R_Vz',
             'Lobe_Temporal_Lz',
             'Lobe_Temporal_Rz',
             'Musc_Constrictz',
             'Parotid_Lz',
             'Parotid_Rz',
             'SpinalCordz',
             'Tracheaz',
             'Thyroidz']
NEW_NAMES = ['Bone_Mandible',
             'BrainStem',
             'Cavity_Oral',
             'Esophagus',
             'Eye_L',
             'Eye_R',
             'Glnd_Submand_L',
             'Glnd_Submand_R',
             'Larynx',
             'LN_L_Ib',
             'LN_L_II-IV',
             'LN_L_V',
             'LN_R_Ib',
             'LN_R_II-IV',
             'LN_R_V',
             'Lobe_Temporal_L',
             'Lobe_Temporal_R',
             'Musc_Constrict',
             'Parotid_L',
             'Parotid_R',
             'SpinalCord',
             'Trachea', 
             'Thyroid']

def make_dicom(dcmfolder, segfolder, label, smooth=False, min_area=5):
    """
    Generates a DICOM RT Structure Set File from binary mask segmentations. To use this,
    first create the protocol structure set in Eclipse, then export the CT and empty
    structure set. This function will fill in the existing structures, but it does not
    create new ones. Any additional filled structures (such as BODY) will be left unchanged.
    
    WARNING: if the CT has been extended in Eclipse by duplicating the end slice, this
    function may not work.

    Parameters
    ----------
    dcmfolder : str
        Folder containing the DICOM CT and RT Structure Set files
    segfolder : str
        Folder containing .mha files with binary mask segmentations of the CT volume
    label : str
        Name for output structure set. Eclipse will use this when it imports.
    smooth : bool, optional
        Apply smoothing to the contours using an elastic surface net approach. Probably not
        necessary as long as you set the Eclipse import filter to keep all structures at
        standard resolution. See surface_net and relax_net below for details. The default is 
        False.
    min_area : float, optional
        Contour areas are checked on each slice, and any small pieces with areas less than
        this value (in mm^2) are discarded. The default is 5.

    """
    allds = [pd.dcmread(os.path.join(dcmfolder, f)) for f in os.listdir(dcmfolder)
             if f.endswith('.dcm')]
    # Load structure set information
    rs = [x for x in allds if x.SOPClassUID == RS_SOP_CLASS_UID][0]
    names = [x.ROIName for x in rs.StructureSetROISequence]
    numbers = [x.ROINumber for x in rs.StructureSetROISequence]
    name_to_num = dict(zip(names, numbers))
    ref_numbers = [x.ReferencedROINumber for x in rs.ROIContourSequence]
    # Load CT information
    ct = [x for x in allds if x.SOPClassUID == CT_SOP_CLASS_UID]
    pixdims = np.array([ct[0].PixelSpacing[0], ct[0].PixelSpacing[1]])
    xvec = np.arange(ct[0].Columns) * pixdims[0] + ct[0].ImagePositionPatient[0]
    yvec = np.arange(ct[0].Rows) * pixdims[1] + ct[0].ImagePositionPatient[1]
    origin = np.array([xvec[0], yvec[0]])
    zvec = [x.SliceLocation for x in ct]
    slice_uids = [x.SOPInstanceUID for x in ct]
    zvec, slice_uids = zip(*sorted(zip(zvec, slice_uids)))
    zvec, slice_uids = np.array(zvec), np.array(slice_uids)
    segfiles = [f for f in os.listdir(segfolder) if f.endswith('.mha')]
    # For each ROI
    for sf in segfiles:
        roiname = NEW_NAMES[[i for i, x in enumerate(OLD_NAMES) if x in sf][0]]
        mask = itk.GetArrayFromImage(itk.ReadImage(os.path.join(segfolder, sf))).astype('u1')
        not_empty = np.array([x.max() > 0 for x in mask])
        index = ref_numbers.index(name_to_num[roiname])
        cont_seq = pd.Sequence()
        rs.ROIContourSequence[index].ContourSequence = cont_seq
        for m, z, uid in zip(mask[not_empty], zvec[not_empty], slice_uids[not_empty]):
            points = get_contours(m, pixdims, origin, min_area)
            if smooth:
                points = [relax_net(p)[::2] for p in points]
            points = [np.vstack((np.around(x, 2).T, len(x)*[z])).T for x in points]
            for p in points:
                cont = pd.Dataset()
                cont_seq.append(cont)
                cont.ContourGeometricType = 'CLOSED_PLANAR'
                cont.ContourData = list(p.flatten())
                cont.NumberOfContourPoints = len(cont)
                cont_img_seq = pd.Sequence()
                cont.ContourImageSequence = cont_img_seq
                cont_img = pd.Dataset()
                cont_img_seq.append(cont_img)
                cont_img.ReferencedSOPClassUID = CT_SOP_CLASS_UID
                cont_img.ReferencedSOPInstanceUID = uid
        print(f'{roiname} completed')
    rs.SOPInstanceUID = pd.uid.generate_uid(UID_PREFIX)
    rs.file_meta.MediaStorageSOPInstanceUID = rs.SOPInstanceUID
    rs.StructureSetLabel = label
    pd.dcmwrite(os.path.join(segfolder, label + '.dcm'), rs)

def get_contours(mask, pix_dims, origin, min_area=5, use4conn=False):
    """
    Finds contour points for the boundaries of regions in a 2D binary mask. Should handle
    handle arbitrary morphology (multiple regions, holes, etc.)

    Parameters
    ----------
    mask : ndarray, dtype=uint8
        The binary mask
    pix_dims : ndarray, dtype=float
        Pixel spacing [x, y]. Typically would get this from the CT files.
    origin : ndarray, dtype=float
        Coordinates of the top left pixel. Typically would get this from teh CT files.
    min_area : float, optional
        Contours with areas less than this value in mm^2 are discarded. The default is 5.
    use4conn : bool, optional
        OpenCV's findContours function uses 8-connectivity. During development I was playing
        around with using 4 connectivity to see if I could improve the results. Ended up not
        being useful. This functionality could be deleted. The default is False.

    Returns
    -------
    list of ndarray of x, y point coordinates for the contours of each region in the mask

    """
    _, contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours = [np.squeeze(x, axis=1) for x in contours]
    # Insert skipped 4-connected points
    if use4conn:
        neighbors = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
        complete = []
        for c in contours:
            distances = np.linalg.norm(np.diff(c, axis=0, append=[c[0]]), axis=1)
            skipped = np.where(distances > 1)[0]
            if skipped.any():
                points = []
                for idx in skipped:
                    start = neighbors.index(tuple(c[idx-1] - c[idx])) // 2
                    search = ((neighbors[::2][start:] + neighbors[::2][:start])[1:] + c[idx])
                    values = mask[tuple(search[:,::-1].T)]
                    points.append(search[np.argmax(values)])
                complete.append(np.insert(c, [x + 1 for x in skipped], points, axis=0))
            else:
                complete.append(c)           
    else:
        complete = contours
    # Convert to DICOM coordinate system and discard small contours
    complete = [x * pix_dims + origin for x in complete]
    areas = [cv.contourArea(x.astype('f4')) for x in complete]
    return [x for x, y in zip(complete, areas) if y > min_area]

def surface_net(mask):
    """
    Create a surface net for a mask slice. The net consists of node coordinates defining the
    region boundaries in the mask. Handles multiple regions and regions with holes. This is
    used for smoothing contours, and is based on the method described in "Constrained Elastic
    SurfaceNets: Generating Smooth MOdels from Binary Segmented Data" (Gibson, 1999).

    Parameters
    ----------
    mask : ndarray of type uint8
        The binary mask.

    Returns
    -------
    complete : list of n x 2 ndarray
        The node coordinates of region boundaries.

    """
    # Identify nodes of 2 x 2 surface squares
    filt = cv.filter2D(mask, -1, np.ones((2,2), 'u1'), borderType=cv.BORDER_CONSTANT)[1:,1:]
    net = np.logical_and(filt != 0, filt != 4).astype('u1')
    # Get contour coordinates in node image
    _, nodes, hierarchy = cv.findContours(net, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    nodes = [np.squeeze(x) for x, y in zip(nodes, hierarchy[0]) if y[3]==-1]
    # Find and insert any 4-connected points missed by findContours
    neighbors = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
    complete = []
    for n in nodes:
        distances = np.linalg.norm(np.diff(n, axis=0, append=[n[0]]), axis=1)
        skipped = np.where(distances > 1)[0]
        if skipped.any():
            points = []
            for idx in skipped:
                start = neighbors.index(tuple(n[idx-1] - n[idx])) // 2
                search = ((neighbors[::2][start:] + neighbors[::2][:start])[1:] + n[idx])
                values = net[tuple(search[:,::-1].T)]
                points.append(search[np.argmax(values)])
            complete.append(np.insert(n, [x + 1 for x in skipped], points, axis=0))
        else:
            complete.append(n)
    return complete
          
def relax_net(nodes, power=3, iterations=10, grid_ext=0.06, grid_num=7):
    """
    Smooths a surface net using an iterative approach that attempts to reduce the distances
    between adjacent nodes while keeping them close to their original locations. This is used
    for smoothing contours, and is based on the method described in "Constrained Elastic
    SurfaceNets: Generating Smooth MOdels from Binary Segmented Data" (Gibson, 1999).

    Parameters
    ----------
    nodes : ndarray
        The surface net.
    power : int, optional
        Determines the strength of the constraint that keeps nodes near their original
        positions. Higher values allow the nodes to move further before the constraint kicks
        in. The default is 3. 
    iterations : int, optional
        Number of relaxations. The default is 10.
    grid_ext : float, optional
        The extent of the search space around each node. The default is 0.06.
    grid_num : TYPE, optional
        The number of grid points along each dimension in the search space around each node.
        The default is 7.

    Returns
    -------
    new_nodes : ndarray
        The relaxed node positions.

    """
    vec = np.linspace(-grid_ext, grid_ext, grid_num)
    xg, yg = np.meshgrid(vec, vec)
    grid = np.vstack((xg.ravel(), yg.ravel()))
    new_nodes = nodes.copy().astype(float)
    anchor = new_nodes.copy()
    idx = list(range(len(anchor)))
    n1 = idx[1:] + idx[:1]
    n2 = idx[-1:] + idx[:-1]
    for _ in range(iterations):
        d1 = new_nodes[..., None] + grid[None, :] - new_nodes[n1][..., None]
        d2 = new_nodes[..., None] + grid[None, :] - new_nodes[n2][..., None]
        d0 = new_nodes[..., None] + grid[None, :] - anchor[..., None]
        cost = (d1**2 + d2**2 + np.abs(d0)**power).sum(axis=1)
        new_nodes += grid.T[cost.argmin(axis=1)]
    return new_nodes

if __name__ == "__main__":
    dcmfolder = '/media/lingshu/data2/HNProsp/Gallagher, John/'
    segfolder = '/media/lingshu/ssd/Head_Neck/Benchmark_Labelmaps/Second_Labelmap/Gallagher, John/' # where the output RTStruct is saved also
    make_dicom(dcmfolder, segfolder, 'AutoTest')