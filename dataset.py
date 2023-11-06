'''Datasets and dataloaders for BOLDMoments and NSD data'''

import os
import numpy as np
import torch
import torch.utils.data as data
from nsd_access import NSDAccess
import pickle as pkl
from matplotlib import pyplot as plt
from PIL import Image



### --------------- NSD Datasets

class NSDImageDataset(data.Dataset):
    """Dataset for NSD returning image data."""

    def __init__(self, 
                 idxs=list(range(0,73000)),
                 nsd_path = '../StableDiffusionReconstruction/nsd',
                 betas_path = '../data/betas_nsd',
                 sub = None,
                 plot = False,
                 resolution = 320,
                 transform = None
                 ):

        self.nsda = NSDAccess(nsd_path)
        self.plot = plot
        self.resolution = resolution
        self.transform = transform


        if sub is not None:
            # Load training pickle
            with open(f'{betas_path}/{sub}/events_imgtag-73k_id.pkl', 'rb') as f:
                d = pkl.load(f)
            self.idxs = [i-1 for i in d[0]] # Substracting 1 because the pkl is 1-indexed, but nsda.read_images expects 0-indexed inputs (goes from 0-72999)
            print(min(self.idxs), max(self.idxs))

        else:
            self.idxs = idxs


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        print("Getting image at NSD index:",self.idxs[idx])

        img = self.nsda.read_images(self.idxs[idx])
        
        
        if self.plot:
            plt.imshow(img)
            plt.show()

        # print('img type after nsda.read_images:', type(img)) # <class 'numpy.ndarray'>
        # print('img shape after nsda.read_images:', img.shape) # (425, 425, 3)
        # print('img min and max after nsda.read_images:', img.min(), img.max()) # 0 255
        img = self.load_img_from_arr_and_transform(img, self.resolution)
        
        if self.transform:
            img = self.transform(img)
                
        return img


    def load_img_from_arr_and_transform(self, img_arr, resolution):
        image = Image.fromarray(img_arr).convert("RGB")
        w, h = resolution, resolution
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        return 2.*image - 1.


class NSDCaptionsDataset(data.Dataset):

    pass
        # if use_captions:
        #     self.captions = self.nsda.read_image_coco_info(idxs, info_type='captions') # Return list of lists of dicts. Each element in the first list is an image. Each element in the second list is a dict for a caption.    
        

class NSDBetasAndTargetsDataset(data.Dataset):

    def __init__(self, 
                 betas_path, 
                 targets_path, 
                 avg_reps=False, 
                 rois=['BMDgeneral'],
                 subs=[1],
                 subset='train'):

        self.betas = []
        self.targets = []
        self.betas_path = betas_path
        self.targets_path = targets_path
        self.rois = rois
        self.subs = subs
        self.subset = subset

        if not avg_reps and subset == 'train':
            repeat_targets = 3
        elif not avg_reps and subset == 'test':
            repeat_targets = 10
        elif avg_reps:
            repeat_targets = 1
        else:
            raise ValueError(f'Unknown subset')
        

        for sub in subs:
            path_to_subject_data = os.path.join(betas_path, f'sub{sub:02d}')

            self.betas.extend(self.load_nsd_betas_impulse(path_to_subject_data, 
                                                          rois, 
                                                          avg_reps=avg_reps,
                                                          subset=subset))

            self.targets.extend(self.load_target_vectors_nsd(targets_path, 
                                                             sub, 
                                                             subset=subset,
                                                             repeat_targets=repeat_targets))

    def __len__(self):
        return len(self.betas)

    def __getitem__(self, idx):

        # Get betas

        # Get target
        target = load_target(self.targets_path, self.targets[idx])

        return self.betas[idx], self.targets[idx]


    def load_target():

        # Load training pickle
        with open(f'{self.betas_path}/sub{subject:02d}/events_imgtag-73k_id.pkl', 'rb') as f:
            img_idxs = pkl.load(f)

    def load_nsd_betas_impulse(self,
                               path_to_subject_data: str, 
                               rois: list, 
                               avg_reps=True,
                               subset='train') -> None:
        
        betas_sub = []

        for roi in rois:
            pkl_name = f'{roi}_betas-GLMsingle_type-typeb_z=1.pkl'

            with open(os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb') as f:
                data = pkl.load(f)

            if avg_reps:
                betas_sub.append(np.mean(data['data_allvoxel'], axis=1))
            else:
                # Concatenate all repetitions into dim 0
                b = np.concatenate([data['data_allvoxel'][:,i,:] for i in range(data['data_allvoxel'].shape[1])])
                betas_sub.append(b)
        
        # TODO: add noise ceiling

        return np.concatenate(betas_sub, axis=1)


    def load_target_vectors_nsd(self,
                                path_to_target_vectors: str, 
                                subject: int, 
                                subset: str = 'train',
                                repeat_targets=1) -> None:
        """
        Load target vectors for a given subject
        """
        targets = []

        # Load training pickle
        with open(f'{self.betas_path}/sub{subject:02d}/events_imgtag-73k_id.pkl', 'rb') as f:
            img_idxs = pkl.load(f)

        train_list = img_idxs[0]
        for target in train_list:
            targets.append(np.load(f'{path_to_target_vectors}/{target-1:06d}.npy'))

        targets = np.array(targets*repeat_targets)

        # flatten vectors
        targets = targets.reshape(targets.shape[0], -1)

        print(f"Loaded {len(img_idxs[0])} NSD img_idxs for subject {subject}, starting with {img_idxs[0][0:5]}")

        return targets


### --------------- BOLDMoments Datasets

class BMDBetasAndTargetsDataset(data.Dataset):
    '''Dataset for BOLDMoments data returning betas and targets.
    
    Can concatenate betas from multiple subjects and ROIs. 
    Returns either train, test or both depending on the subset parameter
    
    Args:
        betas_path (str): path to BOLDMoments betas. 
            This path should point to a folder containing subject subfolders called sub01, sub02, etc.
        targets_path (str): path to target vectors. 
        avg_train_reps (bool): whether to average over repetitions in training data
        beta_type (str): 'impulse' or 'raw'
        rois (list): list of ROIs to include
        subs (list): list of subjects to include
        subset (str): 'train', 'test' or 'both'
    '''
    
    def __init__(self, 
                 betas_path, 
                 targets_path, 
                 avg_reps=False, 
                 beta_type='impulse',
                 rois=['BMDgeneral'],
                 subs=[1],
                 subset='train'):
        
        self.betas = []
        self.targets = []
        self.rois = rois
        self.subs = subs

        if beta_type == 'impulse':
            load_betas = self.load_boldmoments_betas_impulse
        elif beta_type == 'raw':
            load_betas = self.load_boldmoments_betas_raw
        else:
            raise ValueError(f'beta_type must be "impulse" or "raw", not {beta_type}')

        if not avg_reps and subset == 'train':
            repeat_targets = 3
        elif not avg_reps and subset == 'test':
            repeat_targets = 10
        elif avg_reps:
            repeat_targets = 1
        else:
            raise ValueError(f'Unknown subset')

        for sub in subs:
            path_to_subject_data = os.path.join(betas_path, f'sub{sub:02d}')
            
            self.betas.extend(load_betas(path_to_subject_data, 
                                            rois=rois, 
                                            avg_reps=avg_reps,
                                            subset=subset))

            self.targets.extend(self.load_target_vectors_boldmoments(targets_path, 
                                                             subset=subset,
                                                             repeat_targets=repeat_targets))



    def __len__(self):
        return len(self.betas)

    def __getitem__(self, idx):
        return self.betas[idx], self.targets[idx]


    def load_boldmoments_betas_impulse(self,
                                       path_to_subject_data: str, 
                                    rois: list, 
                                    avg_reps: bool = True,
                                    subset: str = 'train',
                                    use_noise_ceiling: bool = True) -> None:
        """
        load betas obtained from assuming the video is an impulse into list that can be used for regression. 
        Loads all the betas for BoldMoments. List should contain N elements, corresponding to the N videos in the selected subset. 
        Each element is an array of betas, concatenating betas for all rois in the roi list.
        """
        betas = []
        
        if subset == 'train':
            key = 'train_data_allvoxel'
            noiseceiling_key = 'train_noiseceiling_allvoxel'
        elif subset == 'test':
            key = 'test_data_allvoxel'
            noiseceiling_key = 'test_noiseceiling_allvoxel'

        for r in rois:
            pkl_name = f'{r}_betas-GLMsingle_type-typed_z=1.pkl'
            with open(os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb') as f:
                data = pkl.load(f)
            
            if avg_reps:
                b = np.mean(data[key], axis=1)
            else:
                # Concatenate all repetitions into dim 0
                b = np.concatenate([data[key][:,i,:] for i in range(data[key].shape[1])])

            # print(b.min(), b.max())
            # print(data[noiseceiling_key]/100)
            # print(data[noiseceiling_key].min(), data[noiseceiling_key].max())
            # print(np.sum(data[noiseceiling_key]/100==0))
            if use_noise_ceiling:
                b = b*(data[noiseceiling_key]/100+0.1) # multiply features by noise ceiling. Add 0.1 to avoid zeroing out 4k features that have a noise ceiling of 0

            # print(b.shape)
            # print(b.min(), b.max())

                
            betas.append(b)

        betas_arr = np.concatenate(betas, axis=1)

        return betas_arr

    def load_boldmoments_betas_raw(self,
                                   path_to_subject_data: str, 
                                   rois: list, 
                                   avg_reps: bool =True,
                                   subset: str ='train') -> None:
        """
        load fMRI data into list that can be used for regression. 
        List should contain N elements, corresponding to the N videos in the selected subset. 
        Each element is an array of betas, concatenating betas for all rois in the roi list.
        """

        betas = []
        key = 'train_data' if subset=='train' else 'test_data'

        for r in rois:
            pkl_name = f"{r}_TRavg-56789_testing.pkl"
            with open(os.path.join(path_to_subject_data, pkl_name), 'rb') as f:
                data = pkl.load(f)

            # Average over repetitions
            if avg_reps:
                betas.append( np.mean(data[key], axis=1))
            else:
                betas.append( data[key])

        betas_arr = np.concatenate(betas, axis=1) 

        return betas_arr


    def load_target_vectors_boldmoments(self,
                                        path_to_target_vectors: str,
                                        subset: str = 'train', 
                                        repeat_targets=1) -> None:
        """
        Load target vectors for a given subject
        """
        targets = []
        if subset == 'train':
            idx_list = list(range(1,1001))
        else:
            idx_list = list(range(1001, 1103))

        for i in idx_list:
            targets.append(np.load(f'{path_to_target_vectors}/{i:04d}.npy'))

        targets_arr = np.array(targets*repeat_targets)

        # flatten vector
        targets_arr = targets_arr.reshape(targets_arr.shape[0], -1)
        
        return targets_arr



### ---- Utility functions

def make_nsd_traintest_for_subj(subj, use_captions=False, use_fmri=False):

    idxs_train, idxs_test = get_nsd_idxs_for_subj(subj) 
    train_dataset = NSDDataset(idxs_train, use_captions=use_captions, use_fmri=use_fmri)   
    test_dataset = NSDDataset(idxs_test, use_captions=use_captions, use_fmri=use_fmri)
    return train_dataset, test_dataset
