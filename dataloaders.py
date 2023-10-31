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
                 idxs=list(range(1,73001)),
                 nsd_path = '../StableDiffusionReconstruction/nsd',
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
            with open(f'data/betas_nsd/{sub}/events_imgtag-73k_id.pkl', 'rb') as f:
                d = pkl.load(f)
            self.idxs = d[0]

        else:
            self.idxs = idxs


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        img = self.nsda.read_images(self.idxs[idx])
        
        if self.plot:
            plt.imshow(img)
            plt.show()

        print('img type after nsda.read_images:', type(img))
        print('img shape after nsda.read_images:', img.shape)
        print('img min and max after nsda.read_images:', img.min(), img.max())
        img = self.load_img_from_arr(img, self.resolution)
        # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        
        if self.transform:
            img = self.transform(img)
        
        return img


    def load_img_from_arr(self, img_arr, resolution):
        image = Image.fromarray(img_arr).convert("RGB")
        w, h = resolution, resolution
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.


    def load_nsd_betas_impulse(path_to_subject_data: str, roi: list, avg_train_reps=True) -> None:
        
        betas_impulse_train_list = []

        for r in roi:
            pkl_name = f'{r}_betas-GLMsingle_type-typeb_z=1.pkl'
            with open(os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb') as f:
                data = pkl.load(f)
            
            if avg_train_reps:
                betas_impulse_train_list.append( np.mean(data['data_allvoxel'], axis=1))
            else:
                # Concatenate all repetitions into dim 0
                data_train = np.concatenate([data['data_allvoxel'][:,i,:] for i in range(data['data_allvoxel'].shape[1])])
                betas_impulse_train_list.append(data_train)

            # TODO: add noise ceiling

        betas_impulse_train_npy = np.concatenate(betas_impulse_train_list, axis=1)

        return betas_impulse_train_npy



    def load_target_vectors_nsd(path_to_target_vectors: str, subject: str, repeat_train=1) -> None:
        """
        Load target vectors for a given subject
        """
        target_train = []
        target_test = []

        # Load training pickle
        with open(f'data/betas_nsd/{subject}/events_imgtag-73k_id.pkl', 'rb') as f:
            img_idxs = pkl.load(f)

        train_list = img_idxs[0]
        for target in train_list:
            target_train.append(np.load(f'{path_to_target_vectors}/{target-1:06d}.npy'))

        target_train = np.array(target_train*repeat_train)

        # flatten vectors
        target_train = target_train.reshape(target_train.shape[0], -1)

        print(f"Loaded {len(img_idxs[0])} NSD img_idxs for subject {subject}, starting with {img_idxs[0][0:5]}")

        return target_train

    def load():
        print(f"Now processing image {s:06} with {len(all_prompts[s-imgidx[0]])} prompts")
        prompts = [p['caption'] for p in all_prompts[s-imgidx[0]]]
        
        img = nsda.read_images(s-imgidx[0])
        if plot:
            from matplotlib import pyplot as plt
            plt.imshow(img)
            plt.title(p['caption'])
            plt.show()
        init_image = load_img_from_arr(img,resolution).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)


class NSDCaptionsDataset(data.Dataset):

    pass
        # if use_captions:
        #     self.captions = self.nsda.read_image_coco_info(idxs, info_type='captions') # Return list of lists of dicts. Each element in the first list is an image. Each element in the second list is a dict for a caption.    
        

class NSDBetasAndTargetsDataset(data.Dataset):

    def __init__():
        if use_fmri:
            # Load the large fMRI pickle containing all betas for a given subject
            subfolder = 'prepared_allvoxel_pkl'
            roi = 'BMDgeneral'
            pkl_name = f'{roi}_betas-GLMsingle_type-typeb_z=1.pkl'
            path_to_subject_data = os.path.join(path_to_betas_nsd, f'sub{sub}')
            with open(os.path.join(path_to_subject_data, subfolder, pkl_name), 'rb') as f:
                data = pkl.load(f)

            self.fmri = data['data_allvoxel']

    def __len__():
        pass

    def __getitem__():

        betas = None
        targets = None

        # for target in train_list:
        #     target_train.append(np.load(f'{path_to_target_vectors}/{target-1:06d}.npy'))

        # target_train = np.array(target_train*repeat_train)

        return betas, targets




### --------------- BOLDMoments Datasets

class BMDDataLoader(data.Dataset):
    pass





def make_nsd_traintest_for_subj(subj, use_captions=False, use_fmri=False):

    idxs_train, idxs_test = get_nsd_idxs_for_subj(subj) 
    train_dataset = NSDDataset(idxs_train, use_captions=use_captions, use_fmri=use_fmri)   
    test_dataset = NSDDataset(idxs_test, use_captions=use_captions, use_fmri=use_fmri)
    return train_dataset, test_dataset
