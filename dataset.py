"""Datasets and dataloaders for BOLDMoments and NSD data"""

import os
import numpy as np
import torch
import torch.utils.data as data
from nsd_access import NSDAccess
import pickle as pkl
from matplotlib import pyplot as plt
from PIL import Image
import scipy
import json
import cv2
import torchvision as tv
from utils import norm_and_transp
import abc




### --------------- Abstract Dataset Classes


class VideoDataset(data.Dataset, abc.ABC):
    """Abstract Dataset class for returning video stimuli."""

    def __init__(self, 
                path, 
                metadata_path = None,
                subset='train',
                resolution=244,
                transform=None,
                normalize=True,
                return_filename=False,
                load_from_frames=False,
                skip_frames=None,
                n_frames_per_video=45,
                minus_1_to_1=True,
                ):
        '''
        Constructor for a generic VideoDataset.
            
    
        Args:
            path (str): path to videos. 
                This path should point to a folder containing the videos/folders that will be listed in vid_paths.
            metadata_path (str): path to HAD metadata
            subset (str): 'train', 'test' or 'all'
            resolution (int): resolution to resize videos to
            transform (torchvision.transforms): transforms to apply to videos
            normalize (bool): whether to normalize and transpose videos to match typical pytorch formats. 
                Applies norm_and_transp() after loading image in uint8 (W,H,C). This fn puts the values in the [-1,1] range and transposes to ensure the dims are (C,W,H)
            return_filename (bool): whether to return the filename of the video
            load_from_frames (bool): whether to load videos frame by frame with cv2.imread instead of with cv2.VideoCapture. If True, path must point to a folder containing folders with the frames of each video.
            skip_frames (int): How many frames to skip between each frame loaded. If None and n_frames_per_video is also None, all frames are loaded. If None and n_frames_per_video is not None, frames are loaded evenly spaced to get n_frames_per_video frames.
            n_frames_per_video (int): How many frames to load per video. If None and skip_frames is also None, all frames are loaded. If None and skip_frames is not None, frames are loaded skipping skip_frames frames, and then padded with the last frame to get n_frames_per_video frames.
        '''

        
        self.path = path
        self.metadata_path = metadata_path
        self.subset = subset
        self.resolution = resolution
        self.transform = transform
        self.normalize = normalize
        self.return_filename = return_filename
        self.load_from_frames = load_from_frames
        self.skip_frames = skip_frames
        self.n_frames_per_video = n_frames_per_video
        self.minus_1_to_1 = minus_1_to_1
        
        if transform is not None:
            self.transform = transform

        self.vid_paths = self.get_video_paths(subset, metadata_path)

    @abc.abstractmethod
    def get_video_paths(self, subset, metadata_path):
        """Returns list of video paths for a given subset"""
        raise NotImplementedError

    def __len__(self):
        return len(self.vid_paths)

    def __getitem__(self, idx):
        if self.load_from_frames:
            video = self.load_frames(idx)
        else:
            video = self.load_video(idx)

        if self.transform:
            video = self.transform(video)
        
        if self.return_filename:
            return video, self.vid_paths[idx]
        return video


    def load_video(self, idx):
        video_path = os.path.join(self.path, self.vid_paths[idx])
        # print("Loading video", video_path)
        if not video_path.endswith('.mp4'):
            video_path = video_path + '.mp4'
        cap = cv2.VideoCapture(video_path)

        # Load frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)

        frames = []
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = len(all_frames)

        frame_indices = self.determine_frame_indices(total_frames)

        # print("Total frames:", total_frames)
        # print("Frame indices:", frame_indices)

        last_frame = None
        for i in frame_indices:
            if i < total_frames:
                frame = self.preprocess_frame(all_frames[i])
                last_frame = frame
            # cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            # ret, frame = cap.read()
            # print("Frame", i, "read:", ret)
            # if ret:
            #     print("Frame shape:", frame.shape)
            #     last_frame = self.preprocess_frame(frame)
            #     frames.append(last_frame)
            # elif last_frame is not None:
            #     frames.append(last_frame)  # Repeat the last frame if needed

        while len(frames) < self.n_frames_per_video:
            frames.append(last_frame)  # Ensure we have n_frames_per_video frames

        cap.release()
        return np.array(frames)

    def load_frames(self, idx):
        """Loads videos frame by frame with cv2.imread and returns a numpy array of frames.
        Requires self.path to be a path to the folder containing whatever is in vid_paths,
        and requires self.vid_paths to be partial paths to frame folders."""

        frames_path = os.path.join(self.path, self.vid_paths[idx])
        frame_files = sorted(os.listdir(frames_path))
        total_frames = len(frame_files)

        frame_indices = self.determine_frame_indices(total_frames)

        last_frame = None
        frames = []
        for i in frame_indices:
            if i < total_frames:
                frame = self.preprocess_frame(cv2.imread(os.path.join(frames_path, frame_files[i])))
                last_frame = frame
            frames.append(last_frame)  # Repeat the last frame if needed

        while len(frames) < self.n_frames_per_video:
            frames.append(last_frame)  # Ensure we have n_frames_per_video frames

        return np.array(frames)

    def determine_frame_indices(self, total_frames):

        if self.n_frames_per_video is not None and self.skip_frames is not None:
            frame_indices = range(0, total_frames, self.skip_frames)
            frame_indices = list(frame_indices)[:self.n_frames_per_video]
        elif self.n_frames_per_video is not None:
            frame_indices = np.linspace(0, total_frames - 1, self.n_frames_per_video, dtype=int)
        elif self.skip_frames is not None:
            frame_indices = range(0, total_frames, self.skip_frames)
        else:
            frame_indices = range(total_frames)

        return frame_indices

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (self.resolution, self.resolution))
        if self.normalize:
            frame_rgb = norm_and_transp(frame_rgb, minus_1_to_1=self.minus_1_to_1)
        return frame_rgb

class ImageDataset(data.Dataset, abc.ABC):
    pass

class CaptionsDataset(data.Dataset, abc.ABC):
    pass

class BetasAndTargetsDataset(data.Dataset, abc.ABC):
    
    def __init__(
        self,
        betas_path,
        targets_path,
        metadata_path,
        bundle_reps=False,
        avg_reps=False,
        rois=["Group41"],
        subs=[1],
        subset="train",
        load_all_in_ram=False,
        use_noise_ceiling=True,
        return_filename=False,
        flatten_targets=True,
    ):
        """
        Constructor for BetasAndTargetsDataset.


        Args:
            betas_path (str): path to betas.
                This path should point to a folder containing subject subfolders called sub01, sub02, etc.
            targets_path (str): path to target vectors, e.g. data/target_vectors_bmd/z_zeroscope
            metadata_path (str): path to metadata
            bundle_reps (bool): If True, returns all repetitions by stacking them in an additional dimension. 
                Needs individual repetition numpys to contain the suffix _rep{rep}.npy. If false, the dataset will return individual repetitions.
            rois (list): list of ROIs to include
            subs (list): list of subjects to include
            subset (str): 'train', 'test' or 'both'
            load_all_in_ram (bool): whether to load all data in RAM. If False, data will be loaded on the fly
            use_noise_ceiling (bool): whether to multiply features by noise ceiling
            return_filename (bool): whether to return the filename of the video
            flatten_targets (bool): whether to flatten target vectors
        """

        assert(~(subset is 'all' and bundle_reps is True and avg_reps is False), 'Cannot returned bundled repetitions when subset is all because of the different number of reps between train and test')

        self.betas = []
        self.targets = []
        self.betas_filenames = []
        self.targets_filenames = []

        self.betas_path = betas_path
        self.targets_path = targets_path
        self.bundle_reps = bundle_reps
        self.avg_reps = avg_reps
        self.rois = rois
        self.subs = subs
        self.subset = subset
        self.load_all_in_ram = load_all_in_ram
        self.use_noise_ceiling = use_noise_ceiling
        self.return_filename = return_filename
        self.flatten_targets = flatten_targets

        self.stim_names = self.get_stim_names(subset, metadata_path)

        self.n_reps = self.train_reps if subset == 'train' else self.test_reps

        for sub in subs:
            if load_all_in_ram:
                self.betas = self.load_all_betas()
                self.targets = self.load_all_target_vectors()
            else:
                self.betas_filenames.extend(self.get_betas_paths(sub))
                self.targets_filenames.extend(self.get_target_paths(sub))

        if self.bundle_reps or self.avg_reps:
            self.reps_per_stim_name = self.get_reps_for_each_stim_name(self.stim_names)


    @property
    @abc.abstractmethod
    def pkl_suffix(self):
        """Returns suffix to compose the pkl name"""
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def train_reps(self):
        """Returns number of repetitions in training data"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_reps(self):
        """Returns number of repetitions in test data"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_stim_names(self, subset, metadata_path):
        """Returns list of stimuli paths for a given subset"""
        raise NotImplementedError

    def get_reps_for_each_stim_name(self, stim_names, dummy_roi='Group41'):
        """Returns a dict with the number of repetitions for each filename"""
        reps_per_filename = {}
        for f in self.betas_filenames:
            f = self.finalize_beta_filename(f, dummy_roi)
            stim_name = os.path.splitext(os.path.basename(f))[0]
            reps_per_filename[stim_name] = []
            for npy_file in os.listdir(os.path.join(self.betas_path, os.path.dirname(f))):
                if stim_name in npy_file:
                    reps_per_filename[stim_name].append(npy_file)
            reps_per_filename[stim_name] = sorted(reps_per_filename[stim_name])
        # print(reps_per_filename)
        return reps_per_filename

    def __len__(self):
        if self.load_all_in_ram:
            return len(self.betas)
        else:
            return len(self.betas_filenames)

    def __getitem__(self, idx):
        if self.load_all_in_ram:
            ret = (self.betas[idx], self.targets[idx])

        else:
            beta = self.load_beta(idx)
            target = self.load_target(idx)
            ret = (beta, target)

        if self.return_filename:
            ret = ret + (self.betas_filenames[idx], self.targets_filenames[idx])

        return ret

    def load_beta(self, idx):
        betas = []
        if self.bundle_reps or self.avg_reps: # Stack reps as an additional dimension. Needs individual repetition numpys to contain the suffix _rep{rep}.npy
            for rep in self.reps_per_stim_name[self.stim_names[idx]]:
                conc_rois = [] 
                for roi in self.rois:
                    beta_filename = self.finalize_beta_filename(self.betas_filenames[idx], roi, rep=rep)
                    conc_rois.append(np.load(os.path.join(self.betas_path, beta_filename)))
                betas.append(np.concatenate(conc_rois))
            if self.avg_reps:
                betas = np.mean(betas, axis=0)
            else:
                betas = np.stack(betas) # betas is a 2-d numpy vector with shape (n_reps, n_betas)
        else:
            for roi in self.rois:
                # print(self.betas_filenames[idx])
                beta_filename = self.finalize_beta_filename(self.betas_filenames[idx], roi)
                betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
            betas = np.concatenate(betas) # betas is a 1-d numpy vector with shape (n_betas,)     
        
        return betas

    def finalize_beta_filename(self, f, roi, rep=None):
        f = f.replace("ROI_FOLDER_PLACEHOLDER", roi+self.pkl_suffix)

        if type(rep) is int:
            f = f.replace(".npy", f"_rep{rep}.npy")
        elif type(rep) is str:
            f = f.replace(os.path.basename(f), rep)
        return f

    def load_target(self, idx):
        target = np.load(os.path.join(self.targets_path, self.targets_filenames[idx]))
        return target.reshape(-1) if self.flatten_targets else target


    def get_betas_paths(self, subject):

        betas_filenames = []
        for n in self.stim_names:
            n = n.split('/')[-1]
            if self.bundle_reps or self.avg_reps:
                betas_filenames.append(
                    os.path.join(
                        f"sub{subject:02d}",
                        "indiv_npys",
                        "ROI_FOLDER_PLACEHOLDER",
                        f"{n}.npy",
                    )
                )
            else:
                for rep in range(self.n_reps):
                    betas_filenames.append(
                        os.path.join(
                            f"sub{subject:02d}",
                            "indiv_npys",
                            "ROI_FOLDER_PLACEHOLDER",
                            f"{n}_rep{rep}.npy",
                        )
                    )
        return betas_filenames

    def get_target_paths(self, subject):
        """Gathers filenames for target vectors."""

        targets_filenames = []

        for n in self.stim_names:
            if self.bundle_reps:
                targets_filenames.append(f"{n}.npy")
            else:
                for _ in range(self.n_reps):
                    targets_filenames.append(f"{n}.npy")

        return targets_filenames


# TODO Clean up

class ReconstructionDataset(data.Dataset, abc.ABC):
    """Abstract dataset returning conditioning vectors (either ground truth or predicted) that can be
    used by a diffusion model (e.g. zeroscope) to reconstruct stimuli."""

    def __init__(
        self,
        cond_vectors_paths_dict,
        metadata_path=None,
        subset="test",
        return_filenames=True,
        rearrange_funcs={},
    ):
        """Constructor for ReconstructionDataset.

        Args:
            cond_vectors_paths_dict (dict): dictionary with keys corresponding
                to the conditioning vector name (e.g. 'z_zeroscope', 'blip' and 'c_zeroscope') and
                values corresponding to the paths to the conditioning vectors folders.
                The folders should contain either large numpy files with all the vectors or individual npys for each vector (e.g. 1001.npy)
            metadata_path (str): path to metadata
            subset (str): 'train', 'test' or 'all'
            return_filenames (bool): whether to return the filename of the conditioning vector
            rearrange_funcs (dict): dictionary with keys corresponding to the conditioning vector name
                and values corresponding to functions to rearrange the conditioning vector. The functions
                should take a 1-d numpy array and return a 1-d numpy array.
        """

        self.cond_vectors_paths_dict = cond_vectors_paths_dict
        self.subset = subset
        self.return_filenames = return_filenames
        self.rearrange_funcs = rearrange_funcs

        self.cond_vectors_to_use = list(cond_vectors_paths_dict.keys())
        self.large_npys = {}
        # Check if given path contains large singular npy files or individual npy files for each stimuli

        for cond_name, path in cond_vectors_paths_dict.items():
            if self.check_if_large_npys(path):
                self.large_npys[cond_name] = np.load(
                    os.path.join(path, f"preds_{subset}.npy")
                )

        self.filenames = self.get_stim_paths(subset, metadata_path)

    def get_stim_paths(self, subset, metadata_path):
        if metadata_path is None: # Assume BMD
            ran = (
                range(1, 1001) if subset == "train" else range(1001, 1103)
            )  # Indexes of train and test sets of BMD
            
            return [f"{i:04d}.npy" for i in ran]


        if subset == 'train':
            self.video_paths = json.load(open(os.path.join(metadata_path,'cc2017_train_set_video_paths.json')))
        elif subset == 'test':
            self.video_paths = json.load(open(os.path.join(metadata_path,'cc2017_test_set_video_paths.json'))) #
        elif subset == 'all':
            self.video_paths_train = json.load(open(os.path.join(metadata_path,'cc2017_train_set_video_paths.json')))
            self.video_paths_test = json.load(open(os.path.join(metadata_path,'cc2017_test_set_video_paths.json')))
            self.video_paths = self.video_paths_train + self.video_paths_test
        else:
            raise ValueError(f'Unknown subset {subset}')
            
        # print('self.video_paths', self.video_paths)
        self.video_paths = sorted(self.video_paths)
        return [f+'.npy' for f in self.video_paths]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        ret = ()
        for cond_name, path in self.cond_vectors_paths_dict.items():
            # print("cond_name", cond_name)
            # print("path", path)
            if cond_name not in self.cond_vectors_to_use:
                continue
            if cond_name in self.large_npys:
                cond_vector = self.large_npys[cond_name][idx]
                is_flat = True
            else:
                cond_vector = np.load(os.path.join(path, self.filenames[idx])).squeeze()
                if cond_vector.ndim == 1:
                    is_flat = True
                else:
                    is_flat = False

            # print(f"{cond_name}.shape before rearrange", cond_vector.shape)
            if cond_name in self.rearrange_funcs and is_flat:
                cond_vector = self.rearrange_funcs[cond_name](cond_vector)

            # DEBUG
            if cond_name == "z" and cond_vector.shape[0] == 15:
                cond_vector = cond_vector.swapaxes(0, 1)
            # print("final cond_vector.shape", cond_vector.shape)
            cond_vector = torch.tensor(cond_vector).float()
            ret = ret + (cond_vector,)

        if self.return_filenames:
            return ret + (self.filenames[idx],)

        return ret

    def get_cond_vector_names(self):
        return self.cond_vectors_paths_dict.keys()

    def set_cond_vectors_to_use(self, cond_vectors_to_use):
        """Allows user to set the exact conditioning vectors that __getitem__ returns by providing a list with names.
        The names must match existing keys in the cond_vectors_paths_dict.

        Args:
            cond_vectors_to_use (list): list of conditioning vector names to use (e.g. ['z', 'blip'])
        """
        self.cond_vectors_to_use = cond_vectors_to_use

    def check_if_large_npys(self, path):
        """Check if given path contains large singular npy files or individual npy files for each stimuli"""
        if os.path.isfile(os.path.join(path, f"preds_{self.subset}.npy")):
            return True
        else:
            return False


### --------------- NSD Datasets


class NSDImageDataset(data.Dataset):
    """Dataset for NSD returning image data."""

    def __init__(
        self,
        idxs=list(range(0, 73000)),
        nsd_path="../StableDiffusionReconstruction/nsd",
        betas_path="../data/betas_nsd",
        sub=None,
        plot=False,
        resolution=320,
        transform=None,
    ):
        self.nsda = NSDAccess(nsd_path)
        self.plot = plot
        self.resolution = resolution
        self.transform = transform

        if sub is not None:
            # Load training pickle
            with open(f"{betas_path}/{sub}/events_imgtag-73k_id.pkl", "rb") as f:
                d = pkl.load(f)
            self.idxs = [
                i - 1 for i in d[0]
            ]  # Substracting 1 because the pkl is 1-indexed, but nsda.read_images expects 0-indexed inputs (goes from 0-72999)
            # print(min(self.idxs), max(self.idxs))

        else:
            self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # print("Getting image at NSD index:",self.idxs[idx])

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
        return 2.0 * image - 1.0


class NSDCaptionsDataset(data.Dataset):
    pass
    # if use_captions:
    #     self.captions = self.nsda.read_image_coco_info(idxs, info_type='captions') # Return list of lists of dicts. Each element in the first list is an image. Each element in the second list is a dict for a caption.


class NSDBetasAndTargetsDataset(data.Dataset):
    pass


class NSDBetasAndTargetsDatasetOld(data.Dataset):
    """
    NSD Dataset returning betas and targets.
    """

    def __init__(
        self,
        betas_path,
        targets_path,
        metadata_path="data/metadata_nsd",
        avg_reps=False,
        rois=["BMDgeneral"],
        subs=[1],
        subset="both",
        load_all_in_ram=False,
        return_filename=False,
        flatten_targets=True,
        num_frames_to_simulate=15,
    ):
        """
        Constructor for NSDBetasAndTargetsDataset.

        Args:
            betas_path (str): path to NSD betas. This path should point to a folder containing subject subfolders called sub01, sub02, etc.
            targets_path (str): Path to target vectors. The folder this path points to should contain 73000 npy arrays corresponding to the target vectors for each NSD stimuli.
            avg_reps (bool): whether to average over repetitions
            rois (list): list of ROIs to include
            subs (list): list of subjects to include
            subset (str): 'train', 'test' or 'both'
        """

        if load_all_in_ram:
            self.betas = []
            self.targets = []
        self.betas_filenames = []
        self.targets_filenames = []

        self.betas_path = betas_path
        self.targets_path = targets_path
        self.avg_reps = avg_reps
        self.rois = rois
        self.subs = subs
        self.subset = subset
        self.load_all_in_ram = load_all_in_ram
        self.return_filename = return_filename
        self.flatten_targets = flatten_targets
        self.num_frames_to_simulate = num_frames_to_simulate
        self.NSD_REPS = 3

        if not avg_reps:
            self.repeat_targets = self.NSD_REPS
        else:
            self.repeat_targets = 1

        nsd_expdesign = scipy.io.loadmat(os.path.join(metadata_path, "nsd_expdesign.mat"))
        test_idxs = nsd_expdesign["sharedix"] - 1

        for sub in subs:
            if load_all_in_ram:
                self.betas.extend(self.load_all_nsd_betas_impulse(sub))
                self.targets.extend(self.load_all_target_vectors_nsd(sub))
            else:
                # Load pickle with ids of stimuli used for this subject
                with open(
                    os.path.join(betas_path, f"sub{sub:02d}/events_imgtag-73k_id.pkl"),
                    "rb",
                ) as f:
                    self.stim_idxs = [
                        i - 1 for i in pkl.load(f)[0]
                    ]  # Substract 1 because the pkl is 1-indexed, but nsda.read_images expects 0-indexed inputs (goes from 0-72999)

                if subset == "train":
                    self.stim_idxs = [i for i in self.stim_idxs if i not in test_idxs]
                elif subset == "test":
                    self.stim_idxs = [i for i in self.stim_idxs if i in test_idxs]

                # print(f"self.stim_idxs len for subset {subset}", len(self.stim_idxs))

                self.betas_filenames.extend(self.gather_betas_impulse_filenames(sub))
                self.targets_filenames.extend(self.gather_target_filenames())

    def __len__(self):
        return len(self.betas_filenames)

    def __getitem__(self, idx):
        if self.load_all_in_ram:
            ret = (self.betas[idx], self.targets[idx])

        else:
            beta = self.load_beta(idx)
            target = self.load_target(idx)
            ret = (beta, target)

        if self.return_filename:
            ret = ret + (self.betas_filenames[idx], self.targets_filenames[idx])

        return ret

    def load_beta(self, idx):
        betas = []
        for roi in self.rois:
            roi_folder = roi + "_betas-GLMsingle_type-typeb_z=1"
            beta_filename = self.betas_filenames[idx].replace(
                "ROI_FOLDER_PLACEHOLDER", roi_folder
            )
            if self.avg_reps:
                # If npy file with average of reps exists, load it. Otherwise, average repetitions on the fly
                if os.path.exists(os.path.join(self.betas_path, beta_filename)):
                    betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
                else:  # Average repetitions on the fly
                    rep = []
                    for r in range(self.NSD_REPS):
                        rep.append(
                            np.load(
                                os.path.join(
                                    self.betas_path, beta_filename[:-4] + f"_{r}.npy"
                                )
                            )
                        )
                    avg_reps = np.mean(rep, axis=0)
                    betas.append(avg_reps)
            else:
                betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
        return np.concatenate(betas)

    def load_target(self, idx):
        target = np.load(os.path.join(self.targets_path, self.targets_filenames[idx]))
        target = np.repeat(target[None], self.num_frames_to_simulate, axis=0)
        if self.flatten_targets:
            return target.reshape(-1)
        return target

    def gather_betas_impulse_filenames(self, sub: int) -> list:
        betas_filenames = []

        for i in self.stim_idxs:
            if self.avg_reps:
                betas_filenames.append(
                    os.path.join(
                        f"sub{sub:02d}",
                        "indiv_npys_avg",
                        "ROI_FOLDER_PLACEHOLDER",
                        f"{i:06d}.npy",
                    )
                )
            else:
                for rep in range(self.repeat_targets):
                    betas_filenames.append(
                        os.path.join(
                            f"sub{sub:02d}",
                            "indiv_npys",
                            "ROI_FOLDER_PLACEHOLDER",
                            f"{i:06d}_{rep}.npy",
                        )
                    )

        return betas_filenames

    def gather_target_filenames(self) -> list:
        targets_filenames = []

        for i in self.stim_idxs:
            for _ in range(self.repeat_targets):
                targets_filenames.append(f"{i:06d}.npy")

        return targets_filenames

    ## Functions to load all in RAM
    def load_all_nsd_betas_impulse(
        self, betas_path: str, sub: int, rois: list, avg_reps=True, subset="train"
    ) -> None:
        subj_betas_path = os.path.join(betas_path, f"sub{sub:02d}")
        betas_sub = []

        for roi in rois:
            pkl_name = f"{roi}_betas-GLMsingle_type-typeb_z=1.pkl"

            with open(
                os.path.join(subj_betas_path, "prepared_allvoxel_pkl", pkl_name), "rb"
            ) as f:
                data = pkl.load(f)

            if avg_reps:
                betas_sub.append(np.mean(data["data_allvoxel"], axis=1))
            else:
                # Concatenate all repetitions into dim 0
                b = np.concatenate(
                    [
                        data["data_allvoxel"][:, i, :]
                        for i in range(data["data_allvoxel"].shape[1])
                    ]
                )
                betas_sub.append(b)

        # TODO: add noise ceiling

        return np.concatenate(betas_sub, axis=1)

    def load_target_vectors_nsd(
        self,
        path_to_target_vectors: str,
        subject: int,
        subset: str = "train",
        repeat_targets=1,
    ) -> None:
        """
        Load target vectors for a given subject
        """
        targets = []

        # Load training pickle
        with open(
            f"{self.betas_path}/sub{subject:02d}/events_imgtag-73k_id.pkl", "rb"
        ) as f:
            img_idxs = pkl.load(f)

        train_list = img_idxs[0]
        for target in train_list:
            targets.append(np.load(f"{path_to_target_vectors}/{target-1:06d}.npy"))

        targets = np.array(targets * repeat_targets)

        # flatten vectors
        targets = targets.reshape(targets.shape[0], -1)

        print(
            f"Loaded {len(img_idxs[0])} NSD img_idxs for subject {subject}, starting with {img_idxs[0][0:5]}"
        )

        return targets


### --------------- BOLDMoments Datasets


class BMDVideoDataset(VideoDataset):
    """Dataset for BMD returning video stimuli."""

    def get_video_paths(self, subset, metadata_path):
        '''Returns list of video paths for a given subset'''

        suffix=''
        if not self.load_from_frames:
            suffix='.mp4'

        if subset == 'train':
            vid_paths = [f'{i:04d}{suffix}' for i in range(1,1001)]
        elif subset == 'test':
            vid_paths = [f'{i:04d}{suffix}' for i in range(1001,1103)]
        elif subset == 'all':
            vid_paths = [f'{i:04d}{suffix}' for i in range(1,1103)]
        else: 
            raise ValueError(f'Unknown subset {subset}')
        
        return vid_paths

class BMDCaptionsDataset(CaptionsDataset):
    pass

class BMDBetasAndTargetsDataset(BetasAndTargetsDataset):

    @property
    def pkl_suffix(self):
        return "_betas-GLMsingle_type-typed_z=1"
    
    @property
    def train_reps(self):
        self.BMD_TRAIN_REPS = 3
        return self.BMD_TRAIN_REPS
    
    @property
    def test_reps(self):
        self.BMD_TEST_REPS = 10
        return self.BMD_TEST_REPS

    def get_stim_names(self, subset, metadata_path):
        # TODO replace this with loading a json in metadata_bmd with the stim names
        if subset == 'train':
            stim_names = [f'{i:04d}' for i in range(1, 1001)]
            self.n_reps = self.train_reps
        elif subset == 'test':
            stim_names = [f'{i:04d}' for i in range(1001, 1103)]
            self.n_reps = self.test_reps
        elif subset == 'all':
            stim_names = [f'{i:04d}' for i in range(1, 1103)]
            self.n_reps = self.train_reps

        return stim_names    

class BMDBetasAndTargetsDatasetOld(data.Dataset):
    """Dataset for BOLDMoments data returning betas and targets.

    Can concatenate betas from multiple subjects and ROIs.
    Returns either train, test or both depending on the subset parameter
    """

    def __init__(
        self,
        betas_path,
        targets_path,
        avg_reps=False,
        beta_type="impulse",
        rois=["BMDgeneral"],
        subs=[1],
        subset="train",
        load_all_in_ram=False,
        use_noise_ceiling=True,
        return_filename=False,
        flatten_targets=True,
    ):
        """
        Constructor for BMDBetasAndTargetsDataset.


        Args:
            betas_path (str): path to BOLDMoments betas.
                This path should point to a folder containing subject subfolders called sub01, sub02, etc.
            targets_path (str): path to target vectors, e.g. data/target_vectors_bmd/z_zeroscope
            avg_train_reps (bool): whether to average over repetitions in training data
            beta_type (str): 'impulse' or 'raw'
            rois (list): list of ROIs to include
            subs (list): list of subjects to include
            subset (str): 'train', 'test' or 'both'
            load_all_in_ram (bool): whether to load all data in RAM. If False, data will be loaded on the fly
            use_noise_ceiling (bool): whether to multiply features by noise ceiling
        """

        self.betas = []
        self.targets = []
        self.betas_filenames = []
        self.targets_filenames = []

        self.betas_path = betas_path
        self.targets_path = targets_path
        self.avg_reps = avg_reps
        self.beta_type = beta_type
        self.rois = rois
        self.subs = subs
        self.subset = subset
        self.load_all_in_ram = load_all_in_ram
        self.use_noise_ceiling = use_noise_ceiling
        self.return_filename = return_filename
        self.flatten_targets = flatten_targets
        self.BMD_TRAIN_REPS = 3
        self.BMD_TEST_REPS = 10

        # BMD's stimuli indexes are 1-1000 if train, 1001-1102 if test
        if subset == "train":
            self.stim_idxs = list(range(1, 1001))
        elif subset == "test":
            self.stim_idxs = list(range(1001, 1103))

        if not avg_reps and subset == "train":
            self.repeat_targets = self.BMD_TRAIN_REPS
        elif not avg_reps and subset == "test":
            self.repeat_targets = self.BMD_TEST_REPS
        elif avg_reps:
            self.repeat_targets = 1
        else:
            raise ValueError(f"Unknown subset")

        for sub in subs:
            if load_all_in_ram:  # TODO FINISH THIS
                if beta_type == "impulse":
                    load_all_betas_fn = self.load_all_boldmoments_betas_impulse
                elif beta_type == "raw":
                    load_all_betas_fn = self.load_all_boldmoments_betas_raw
                else:
                    raise ValueError(
                        f'beta_type must be "impulse" or "raw", not {beta_type}'
                    )

                path_to_subject_data = os.path.join(betas_path, f"sub{sub:02d}")

                self.betas.extend(
                    load_all_betas_fn(
                        path_to_subject_data,
                        rois=rois,
                        avg_reps=avg_reps,
                        subset=subset,
                    )
                )

                self.targets.extend(
                    self.load_all_target_vectors_boldmoments(
                        targets_path, subset=subset, repeat_targets=self.repeat_targets
                    )
                )
            else:
                self.betas_filenames.extend(self.gather_betas_impulse_filenames(sub))
                self.targets_filenames.extend(self.gather_target_filenames())

    def __len__(self):
        if self.load_all_in_ram:
            return len(self.betas)
        else:
            return len(self.betas_filenames)

    def __getitem__(self, idx):
        if self.load_all_in_ram:
            ret = (self.betas[idx], self.targets[idx])

        else:
            beta = self.load_beta(idx)
            target = self.load_target(idx)
            ret = (beta, target)

        if self.return_filename:
            ret = ret + (self.betas_filenames[idx], self.targets_filenames[idx])

        return ret

    def load_beta(self, idx):
        betas = []
        for roi in self.rois:
            roi_folder = roi + "_betas-GLMsingle_type-typed_z=1"
            beta_filename = self.betas_filenames[idx].replace(
                "ROI_FOLDER_PLACEHOLDER", roi_folder
            )

            if self.avg_reps:
                # If npy file with average of reps exists, load it. Otherwise, average repetitions on the fly
                if os.path.exists(os.path.join(self.betas_path, beta_filename)):
                    betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
                else:  # Average repetitions on the fly
                    rep = []
                    subs_reps = (
                        self.BMD_TRAIN_REPS
                        if self.subset == "train"
                        else self.BMD_TEST_REPS
                    )
                    for r in range(subs_reps):
                        rep.append(
                            np.load(
                                os.path.join(
                                    self.betas_path, beta_filename[:-4] + f"_rep{r}.npy"
                                )
                            )
                        )
                    avg_reps = np.mean(rep, axis=0)
                    betas.append(avg_reps)

            else:
                betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
        return np.concatenate(betas)

    def load_target(self, idx):
        target = np.load(os.path.join(self.targets_path, self.targets_filenames[idx]))
        if self.flatten_targets:
            return target.reshape(-1)
        return target

    def gather_betas_impulse_filenames(self, sub: int) -> list:
        betas_filenames = []

        for i in self.stim_idxs:
            if self.avg_reps:
                betas_filenames.append(
                    os.path.join(
                        f"sub{sub:02d}",
                        "indiv_npys_avg",
                        "ROI_FOLDER_PLACEHOLDER",
                        f"{i:04d}.npy",
                    )
                )
            else:
                for rep in range(self.repeat_targets):
                    betas_filenames.append(
                        os.path.join(
                            f"sub{sub:02d}",
                            "indiv_npys",
                            "ROI_FOLDER_PLACEHOLDER",
                            f"{i:04d}_rep{rep}.npy",
                        )
                    )
        return betas_filenames

    def gather_target_filenames(self) -> list:
        """Gathers filenames for target vectors.
        Assumes that target vectors are named 0001.npy, 0002.npy, etc."""

        targets_filenames = []

        for i in self.stim_idxs:
            for _ in range(self.repeat_targets):
                targets_filenames.append(f"{i:04d}.npy")

        return targets_filenames

    ## Functions to load all in RAM
    def load_all_boldmoments_betas_impulse(
        self,
        path_to_subject_data: str,
        rois: list,
        avg_reps: bool = True,
        subset: str = "train",
        use_noise_ceiling: bool = True,
    ) -> None:
        """
        load betas obtained from assuming the video is an impulse into list that can be used for regression.
        Loads all the betas for BoldMoments. List should contain N elements, corresponding to the N videos in the selected subset.
        Each element is an array of betas, concatenating betas for all rois in the roi list.
        """
        betas = []

        if subset == "train":
            key = "train_data_allvoxel"
            noiseceiling_key = "train_noiseceiling_allvoxel"
        elif subset == "test":
            key = "test_data_allvoxel"
            noiseceiling_key = "test_noiseceiling_allvoxel"

        for r in rois:
            pkl_name = f"{r}_betas-GLMsingle_type-typed_z=1.pkl"
            with open(
                os.path.join(path_to_subject_data, "prepared_allvoxel_pkl", pkl_name),
                "rb",
            ) as f:
                data = pkl.load(f)

            if avg_reps:
                b = np.mean(data[key], axis=1)
            else:
                # Concatenate all repetitions into dim 0
                b = np.concatenate(
                    [data[key][:, i, :] for i in range(data[key].shape[1])]
                )

            # print(b.min(), b.max())
            # print(data[noiseceiling_key]/100)
            # print(data[noiseceiling_key].min(), data[noiseceiling_key].max())
            # print(np.sum(data[noiseceiling_key]/100==0))
            if use_noise_ceiling:
                b = (
                    b * (data[noiseceiling_key] / 100 + 0.1)
                )  # multiply features by noise ceiling. Add 0.1 to avoid zeroing out 4k features that have a noise ceiling of 0

            # print(b.shape)
            # print(b.min(), b.max())

            betas.append(b)

        betas_arr = np.concatenate(betas, axis=1)

        return betas_arr

    def load_boldmoments_betas_raw(
        self,
        path_to_subject_data: str,
        rois: list,
        avg_reps: bool = True,
        subset: str = "train",
    ) -> None:
        """
        load fMRI data into list that can be used for regression.
        List should contain N elements, corresponding to the N videos in the selected subset.
        Each element is an array of betas, concatenating betas for all rois in the roi list.
        """

        betas = []
        key = "train_data" if subset == "train" else "test_data"

        for r in rois:
            pkl_name = f"{r}_TRavg-56789_testing.pkl"
            with open(os.path.join(path_to_subject_data, pkl_name), "rb") as f:
                data = pkl.load(f)

            # Average over repetitions
            if avg_reps:
                betas.append(np.mean(data[key], axis=1))
            else:
                betas.append(data[key])

        betas_arr = np.concatenate(betas, axis=1)

        return betas_arr

    def load_all_target_vectors_boldmoments(
        self, path_to_target_vectors: str, subset: str = "train", repeat_targets=1
    ) -> None:
        """
        Load target vectors for a given subject
        """
        targets = []
        if subset == "train":
            idx_list = list(range(1, 1001))
        else:
            idx_list = list(range(1001, 1103))

        for i in idx_list:
            targets.append(np.load(f"{path_to_target_vectors}/{i:04d}.npy"))

        targets_arr = np.array(targets * repeat_targets)

        # flatten vector
        targets_arr = targets_arr.reshape(targets_arr.shape[0], -1)

        return targets_arr


### --------------- HAD Datasets


class HADVideoDataset(VideoDataset):
    """Dataset for HAD returning video stimuli."""

    def get_video_paths(self, subset, metadata_path):
        """Returns list of video paths for a given subset"""
        if subset == "train":
            video_paths = json.load(
                open(os.path.join(metadata_path, "had_train_set_video_paths.json"))
            )
        elif subset == "test":
            video_paths = json.load(
                open(os.path.join(metadata_path, "had_test_set_video_paths.json"))
            )
        elif subset == "all":
            video_paths_train = json.load(
                open(os.path.join(metadata_path, "had_train_set_video_paths.json"))
            )
            video_paths_test = json.load(
                open(os.path.join(metadata_path, "had_test_set_video_paths.json"))
            )
            video_paths = video_paths_train + video_paths_test
        else:
            raise ValueError(f"Unknown subset {subset}")

        self.vid_paths = video_paths
        return video_paths


class HADBetasAndTargetsDataset(BetasAndTargetsDataset):

    @property
    def pkl_suffix(self):
        return "_betas-GLMsingle_type-typeb_z=1"
    
    @property
    def train_reps(self):
        self.HAD_TRAIN_REPS = 1
        return self.HAD_TRAIN_REPS

    @property
    def test_reps(self):
        self.HAD_TEST_REPS = 1
        return self.HAD_TEST_REPS

    def get_stim_names(self, subset, metadata_path):
        """Returns list of stimuli paths for a given subset"""

        video_paths = []
        if subset == "train":
            train_set = json.load(
                open(os.path.join(metadata_path, "had_train_set_per_subject.json"))
            )
            for sub in self.subs:
                video_paths.extend(train_set[f'sub{sub:02d}'])
        elif subset == "test":
            test_set = json.load(
                open(os.path.join(metadata_path, "had_test_set_per_subject.json"))
            )
            for sub in self.subs:
                video_paths.extend(test_set[f'sub{sub:02d}'])
        elif subset == "all":
            train_set = json.load(
                open(os.path.join(metadata_path, "had_train_set_per_subject.json"))
            )
            test_set = json.load(
                open(os.path.join(metadata_path, "had_test_set_per_subject.json"))
            )
            for sub in self.subs:
                video_paths.extend(train_set[f'sub{sub:02d}'])
                video_paths.extend(test_set[f'sub{sub:02d}'])

        else:
            raise ValueError(f"Unknown subset {subset}")

        stim_names = []
        for v in video_paths:
            # v = v.split('/')[-1]
            if v.endswith('.mp4'):
                v = v[:-4]
            stim_names.append(v)
                
        return stim_names        


class HADBetasAndTargetsDatasetOld(data.Dataset):
    """Dataset for HAD data returning betas and targets.

    Can concatenate betas from multiple subjects and ROIs.
    Returns either train, test or both depending on the subset parameter
    """

    def __init__(
        self,
        betas_path,
        targets_path,
        metadata_path="../data/metadata_had",
        avg_reps=False,
        beta_type="impulse",
        rois=["Group41"],
        subs=[1],
        subset="train",
        load_all_in_ram=False,
        use_noise_ceiling=True,
        return_filename=False,
        flatten_targets=True,
    ):
        """
        Constructor for HADBetasAndTargetsDataset.


        Args:
            betas_path (str): path to HAD betas.
                This path should point to a folder containing subject subfolders called sub01, sub02, etc.
            targets_path (str): path to target vectors, e.g. data/target_vectors_bmd/z_zeroscope
            avg_train_reps (bool): whether to average over repetitions in training data
            beta_type (str): 'impulse' or 'raw'
            rois (list): list of ROIs to include
            subs (list): list of subjects to include
            subset (str): 'train', 'test' or 'both'
            load_all_in_ram (bool): whether to load all data in RAM. If False, data will be loaded on the fly
            use_noise_ceiling (bool): whether to multiply features by noise ceiling
        """

        self.betas = []
        self.targets = []
        self.betas_filenames = []
        self.targets_filenames = []

        self.betas_path = betas_path
        self.targets_path = targets_path
        self.avg_reps = avg_reps
        self.beta_type = beta_type
        self.rois = rois
        self.subs = subs
        self.subset = subset
        self.load_all_in_ram = load_all_in_ram
        self.use_noise_ceiling = use_noise_ceiling
        self.return_filename = return_filename
        self.flatten_targets = flatten_targets

        if subset == "train":
            self.stim_names_to_use = json.load(
                open(os.path.join(metadata_path, "had_train_set_video_paths.json"))
            )
        elif subset == "test":
            self.stim_names_to_use = json.load(
                open(os.path.join(metadata_path, "had_test_set_video_paths.json"))
            )

        for sub in subs:
            if load_all_in_ram:
                # TODO Finish
                self.betas = self.load_all_had_betas()
                self.targets = self.load_all_target_vectors_had()
            else:
                # self.betas_filenames.extend(self.gather_betas_impulse_filenames(sub))
                self.betas_filenames.extend(self.gather_betas_filenames(sub))
                self.targets_filenames.extend(self.gather_target_filenames())

    def __len__(self):
        return len(self.betas) if self.load_all_in_ram else len(self.betas_filenames)

    def __getitem__(self, idx):
        if self.load_all_in_ram:
            ret = (self.betas[idx], self.targets[idx])

        else:
            beta = self.load_beta(idx)
            target = self.load_target(idx)
            ret = (beta, target)

        if self.return_filename:
            ret = ret + (self.betas_filenames[idx], self.targets_filenames[idx])

        return ret

    def load_beta(self, idx):
        betas = []
        for roi in self.rois:
            roi_folder = f"{roi}_betas-GLMsingle_type-typeb_z=1"
            beta_filename = self.betas_filenames[idx].replace(
                "ROI_FOLDER_PLACEHOLDER", roi_folder
            )

            betas.append(np.load(os.path.join(self.betas_path, beta_filename)))
        return np.concatenate(betas)

    def load_target(self, idx):
        target = np.load(os.path.join(self.targets_path, self.targets_filenames[idx]))
        return target.reshape(-1) if self.flatten_targets else target

    def gather_betas_filenames(self, sub: int) -> list:
        """
        Gathers beta filenames for the HAD dataset. Requires self.stim_names_to_use to be defined.
        self.stim_names_to_use must be a list of HAD video names to use in this dataset,
        typically a list of training videos or a list of test videos. Stim name example: v_Archery_id_9FFEroHG-fY_start_59.5_label_1
        """
        betas_filenames = []
        for s in self.stim_names_to_use:  # Stim names to use must be a list of the HAD video names to use in this dataset, typically a list of training videos or a list of test videos. Stim name example: 'v_Archery_id_9FFEroHG-fY_start_59.5_label_1'
            s = s.split("/")[-1].replace(".mp4", ".npy")
            betas_filenames.append(
                os.path.join(f"sub{sub:02d}", "indiv_npys", "ROI_FOLDER_PLACEHOLDER", s)
            )
        return betas_filenames

    def gather_target_filenames(self):
        targets_filenames = []
        for s in self.stim_names_to_use:
            targets_filenames.append(s)
        return targets_filenames

    ## Functions to load all in RAM
    def load_all_had_betas(self):
        """
        Load all HAD betas in RAM.
        Returns a list of betas, where each element is an array of betas for a given subject and ROI.
        """
        betas = []
        raise NotImplementedError

    def load_all_target_vectors_had(self):
        """
        Load all HAD target vectors in RAM.
        Returns a list of target vectors, where each element is a target vector for a given subject.
        """
        targets = []
        raise NotImplementedError


### --------------- NOD Datasets


class NODImageDataset(ImageDataset):
    """Dataset for NOD returning image stimuli."""

    def get_video_paths(self, subset, metadata_path):
        """Returns list of video paths for a given subset"""
        if subset == "train":
            self.video_paths = json.load(
                open(os.path.join(metadata_path, "nod_train_set_video_paths.json"))
            )
        elif subset == "test":
            self.video_paths = json.load(
                open(os.path.join(metadata_path, "nod_test_set_video_paths.json"))
            )
        elif subset == "all":
            video_paths_train = json.load(
                open(os.path.join(metadata_path, "nod_train_set_video_paths.json"))
            )
            video_paths_test = json.load(
                open(os.path.join(metadata_path, "nod_test_set_video_paths.json"))
            )
            self.video_paths = video_paths_train + video_paths_test
        else:
            raise ValueError(f"Unknown subset {subset}")


        return self.video_paths


class NODBetasAndTargetsDataset(data.Dataset):
    pass

### --------------- CC2017 Datasets


class CC2017VideoDataset(VideoDataset):
    '''Dataset for CC2017 returning video stimuli.'''

    def get_video_paths(self, subset, metadata_path):

        if subset == 'train':
            self.video_paths = json.load(open(os.path.join(metadata_path,'cc2017_train_set_video_paths.json')))
        elif subset == 'test':
            self.video_paths = json.load(open(os.path.join(metadata_path,'cc2017_test_set_video_paths.json')))
        elif subset == 'all':
            self.video_paths_train = json.load(open(os.path.join(metadata_path,'cc2017_train_set_video_paths.json')))
            self.video_paths_test = json.load(open(os.path.join(metadata_path,'cc2017_test_set_video_paths.json')))
            self.video_paths = self.video_paths_train + self.video_paths_test
        else:
            raise ValueError(f'Unknown subset {subset}')
            
        # print('self.video_paths', self.video_paths)
        return self.video_paths


class CC2017BetasAndTargetsDataset(BetasAndTargetsDataset):

    @property
    def pkl_suffix(self):
        return "_estimates-TSTrialEstimates_z=1"

    @property
    def train_reps(self):
        self.CC2017_TRAIN_REPS = 2
        return self.CC2017_TRAIN_REPS
    
    @property
    def test_reps(self):
        self.CC2017_TEST_REPS = 10
        return self.CC2017_TEST_REPS

    def get_stim_names(self, subset, metadata_path):
        if subset == 'train':
            stim_names = json.load(open(os.path.join(metadata_path, 'cc2017_train_set_video_paths.json')))
        elif subset == 'test':
            stim_names = json.load(open(os.path.join(metadata_path, 'cc2017_test_set_video_paths.json')))
        elif subset == 'all':
            stim_names_train = json.load(open(os.path.join(metadata_path, 'cc2017_train_set_video_paths.json')))
            stim_names_test = json.load(open(os.path.join(metadata_path, 'cc2017_test_set_video_paths.json')))
            stim_names = stim_names_train + stim_names_test
        else:
            raise ValueError(f'Unknown subset {subset}')
        
        if stim_names[0].endswith('.mp4'):
            stim_names = [s[:-4] for s in stim_names]

        self.stim_names = stim_names

        return self.stim_names
    


### ---- Datasets for Reconstruction

class CC2017ReconstructionDataset(data.Dataset):
    pass #TODO Finish

    


# class NSDReconstructionDataset():
#     '''Dataset for NSD data returning conditioning vectors (either ground truth or predicted) that can be
#     used by a diffusion model (e.g. zeroscope) to reconstruct the NSD images.'''

#     def __init__(self,
#                  cond_vectors_paths_dict,
#                  subset='test',
#                  return_filenames=True,
#                  rearrange_expressions=None):
#         '''Constructor for NSDReconstructionDataset.

#         Args:
#             cond_vectors_paths_dict (dict): dictionary with keys corresponding
#                 to the conditioning vector name (e.g. 'z', 'blip' and 'c') and
#                 values corresponding to the paths to the conditioning vectors folders.
#                 The folders should either contain a large npy file with all the vectors
#                 for a subset of NSD (e.g. preds_train) or a set of individual npy files for each NSD image (e.g. 1001.npy)
#             subset (str): 'train' or 'test'
#             return_filenames (bool): whether to return the filename of the conditioning vector
#         '''

#         self.cond_vectors_paths_dict = cond_vectors_paths_dict
#         self.subset = subset
#         self.return_filenames = return_filenames

#         self.cond_vectors_to_use = list(cond_vectors_paths_dict.keys())
#         self.large_npys = {}
#         # Check if given path contains large singular npy files or individual npy files for each stimuli

#         for cond_name, path in cond_vectors_paths_dict.items():
#             if self.check_if_large_npys(path):
#                 self.large_npys[cond_name] = np.load(os.path.join(path, f'preds_{subset}.npy'))

#         ran = range(0,73000) # Indexes of train and test sets of NSD
#         self.filenames = [f'{i:06d}.npy' for i in ran]


#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):

#         ret = ()
#         for cond_name, path in self.cond_vectors_paths_dict.items():
#             if cond_name not in self.cond_vectors_to_use:
#                 continue
#             if cond_name in self.large_npys:
#                 cond_vector = self.large_npys[cond_name][idx]
#                 flattened_vecs = True
#             else:
#                 cond_vector = np.load(os.path.join(path, self.filenames[idx]))
#                 if cond_vector.ndim == 1:
#                     flattened_vecs = True
#             if self.rearrange_expressions is not None and cond_name in self.rearrange_expressions and flattened_vecs:
#                 cond_vector = rearrange(cond_vector, self.rearrange_expressions[cond_name])

#         if self.return_filenames:
#             return ret + (self.filenames[idx],)
#         return ret


### ---- Utility functions

# def make_nsd_traintest_for_subj(subj, use_captions=False, use_fmri=False):
#     idxs_train, idxs_test = get_nsd_idxs_for_subj(subj)
#     train_dataset = NSDDataset(idxs_train, use_captions=use_captions, use_fmri=use_fmri)
#     test_dataset = NSDDataset(idxs_test, use_captions=use_captions, use_fmri=use_fmri)
#     return train_dataset, test_dataset




### -------------- Maps

# TODO transform into config files for each dataset, to make it easy to add a new one
VIDEO_DATASET_MAP = {
    'bmd': BMDVideoDataset,
    'bmd_captions': BMDCaptionsDataset,
    'had': HADVideoDataset,
    'nsd': NSDImageDataset,
    'nod': NODImageDataset,
    'cc2017': CC2017VideoDataset,
}

BT_DATASET_MAP = {
    'bmd': BMDBetasAndTargetsDatasetOld,
    'bmd_captions': BMDBetasAndTargetsDatasetOld,
    'had': HADBetasAndTargetsDatasetOld,
    'nsd': NSDBetasAndTargetsDataset,
    'nod': NODBetasAndTargetsDataset,
    'cc2017': CC2017BetasAndTargetsDataset,
}

DATASET_OUTPUT_TYPES = {
    'bmd': 'video',
    'bmd_captions': 'text',
    'had': 'video',
    'nsd': 'image',
    'nod': 'image',
    'cc2017': 'video',
} 

DATASET_PATHS = {
    'bmd': 
        {'stimuli': './data/stimuli_bmd/mp4', 
        'metadata': './data/metadata_bmd',
        'betas': './data/betas_cifti_bmd',
        'targets': './data/target_vectors_bmd'
        },
    'bmd_captions': 
        {'stimuli': './data/stimuli_bmd/captions', 
        'metadata': './data/metadata_bmd_captions',
        'betas': './data/betas_cifti_bmd',
        'targets': './data/target_vectors_bmd_captions'
        },
    'had': 
        {'stimuli':'./data/stimuli_had', 
        'metadata':'./data/metadata_had',
        'betas':'./data/betas_cifti_had',
        'targets':'./data/target_vectors_had'
        },
    'nsd': 
        {'stimuli': './data/stimuli_nsd', 
        'metadata': './data/metadata_nsd',
        'betas': './data/betas_cifti_nsd',
        'targets': './data/target_vectors_nsd'
        },
    'nod': 
        {'stimuli': './data/stimuli_nod', 
        'metadata': './data/metadata_nod',
        'betas': './data/betas_cifti_nod',
        'targets': './data/target_vectors_nod'
        },
    'cc2017': 
        {'stimuli': './data/stimuli_cc2017', 
        'metadata': './data/metadata_cc2017',
        'betas': './data/betas_cifti_cc2017',
        'targets': './data/target_vectors_cc2017'
        },
}