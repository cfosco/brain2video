import os
import numpy as np
import pickle

class arguments():
    def __init__(self) -> None:
        self.root = "/mnt/t/BMDGeneration/analysis/GLMMNI152_impulse/sub-01/GLMsingle/betas-prepared/prepared_allvoxel_pkl" #your path here

args = arguments()

xyz = (78, 93, 71) #size of complete volume
mapping_flattened = np.ones(np.prod(xyz)) * -99 #used to map between full volume and bmdgeneral (or any other roi)

#load BMDgeneral ROI
with open(os.path.join(args.root, "BMDgeneral_betas-GLMsingle_type-typed_z=1.pkl"), 'rb') as f:
    bmdgeneral_dict = pickle.load(f)
bmdgeneral_train = bmdgeneral_dict['train_data_allvoxel']
bmdgeneral_test = bmdgeneral_dict['test_data_allvoxel']

mapping_flattened[bmdgeneral_dict['roi_indices_fullvolume']] = np.arange(len(bmdgeneral_dict['roi_indices_fullvolume'])) #creates a mapping between full volume and roi indices

#load ROIs you want to remove from BMDgeneral e.g., FFA, OFA, STS, EBA
rois_to_remove = ['FFA', 'OFA', 'STS', 'EBA']
for roi in rois_to_remove:
    for hemi in ['l','r']:
        with open(os.path.join(args.root, f"{hemi}{roi}_betas-GLMsingle_type-typed_z=1.pkl"), 'rb') as f:
            roi_dict = pickle.load(f)
        roi_indices_mapped = mapping_flattened[roi_dict['roi_indices_fullvolume']]
        indices_to_ablate = roi_indices_mapped[roi_indices_mapped > -1].astype(int) #don't want the -99 indices
        
        bmdgeneral_train[:,:,indices_to_ablate] = 0 #or change to any value
        bmdgeneral_test[:,:,indices_to_ablate] = 0 #or change to any value

