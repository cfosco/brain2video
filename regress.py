import argparse, os
import numpy as np
import pickle as pkl
from utils import compute_metrics
from himalaya.backend import set_backend
from torch.utils.data import ConcatDataset, DataLoader
import yaml
import wandb

from einops import rearrange
from tqdm import tqdm

from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import NSDBetasAndTargetsDataset, BMDBetasAndTargetsDataset
from models import MLPRegressor, SwiGLURegressor, HimalayaRidgeRegressor

def main(args):
    # Load yaml config
    # with open(args.config, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #     print(config)

    config = args

    # Weights and biases setup
    # wandb.login()
    # wandb.init(project="brain2video",
    #             config=config)

    backend = set_backend("numpy")  # or "torch_cuda"

    if config.avg_train_reps:
        repeat_train = 1
    else:
        repeat_train = 3

    ## Paths to fMRI data (our input features) and vectors to regress (our output targets)
    nsd_betas_path = f'data/betas_nsd'
    nsd_targets_path = f'data/target_vectors_nsd/{config.target}'
    bmd_betas_path = f'data/{config.betas_type}'
    bmd_targets_path = f'data/target_vectors/{config.target}'

    ## Build method string
    method = f'regressor:{config.regressor}withscheduleronval_hidden:{config.hidden_size}_fmritype:{config.betas_type}_rois:{"-".join(config.roi)}_avgtrainreps:{config.avg_train_reps}_usensd:{config.use_nsd}'
    print("Method:", method)

    # If pretrain NSD, load NSD data
    if args.use_nsd:

        nsd_dataset_both = NSDBetasAndTargetsDataset(
                        betas_path=nsd_betas_path,
                        targets_path=nsd_targets_path,
                        avg_reps=False, 
                        rois=config.roi,
                        subs=config.nsd_sub,
                        subset='both',
                        load_all_in_ram=False,
                        num_frames_to_simulate=1 if config.target=='blip' else 15)

        # fmri_feat_train_nsd = load_nsd_betas_impulse(
        #     f'data/betas_nsd/{subject}', roi=roi, avg_train_reps=config.avg_train_reps
        # )
        # target_train_nsd = load_target_vectors_nsd(
        #     f'data/target_vectors_nsd/{target}',
        #     subject=subject,  # NSD has different targets per subject, as not all subjects saw all stimuli
        #     repeat_train=repeat_train,
        # )

    bmd_dataset_train = BMDBetasAndTargetsDataset(
                        bmd_betas_path,
                        bmd_targets_path,
                        avg_reps=False, 
                        rois=config.roi,
                        subs=config.bmd_sub,
                        subset='train',
                        load_all_in_ram=False)
    
    bmd_dataset_test = BMDBetasAndTargetsDataset(
                        bmd_betas_path,
                        bmd_targets_path,
                        avg_reps=False, 
                        rois=config.roi,
                        subs=config.bmd_sub,
                        subset='test',
                        load_all_in_ram=False)
    
    ## Load train and test input features
    # if fmri_type == 'betas_impulse':
    #     fmri_feat_train, fmri_feat_test = load_boldmoments_betas_impulse(
    #         fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
    #     )
    # elif fmri_type == 'betas_raw':
    #     fmri_feat_train, fmri_feat_test = load_boldmoments_betas_raw(
    #         fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
    #     )

    ## Load train and test output targets
    # target_train, target_test = load_target_vectors_boldmoments(
    #     targets_path, repeat_train=repeat_train
    # )

    # Concatenate NSD data
    if args.use_nsd:

        dataset_train = ConcatDataset([nsd_dataset_both, bmd_dataset_train])
        dataset_train.subset = 'train'
        dataset_train.return_filename = False
        print("Instantiated Train dataset as concatenation of NSD and BMD datasets")
        print("Length of nsd_dataset_both:", len(nsd_dataset_both))
        print("Length of bmd_dataset_train:", len(bmd_dataset_train))
        print("Length of concatenated dataset:", len(dataset_train))
        # print("Shapes before concatenating NSD data:")
        # print(f"fmri_feat_train: {fmri_feat_train.shape}")
        # print(f"target_train: {target_train.shape}")
        # print(f"fmri_feat_train_nsd: {fmri_feat_train_nsd.shape}")
        # print(f"target_train_nsd: {target_train_nsd.shape}")
        # fmri_feat_train = np.concatenate([fmri_feat_train, fmri_feat_train_nsd], axis=0)
        # target_train = np.concatenate([target_train, target_train_nsd], axis=0)

        # print("Shapes after concatenating NSD data:")
        # print(f"fmri_feat_train: {fmri_feat_train.shape}")
        # print(f"target_train: {target_train.shape}")
    else:
        dataset_train = bmd_dataset_train
    
    dataset_test = bmd_dataset_test
    print("Instantiated Test dataset as BMD dataset")
    print("Length of bmd_dataset_test:", len(bmd_dataset_test))


    ## Define regressor
    if config.regressor == 'himalaya-ridge':
        pipeline = make_pipeline_for_himalaya_regressor(config, backend)
    elif config.regressor == 'mlp':
        # input_shape = fmri_feat_train.shape[1]
        # output_shape = target_train.shape[1]
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = MLPRegressor(input_shape, output_shape, hidden_size=config.hidden_size).to('cuda')
    elif config.regressor == 'swiglu':
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = SwiGLURegressor(input_shape, output_shape, hidden_features=config.hidden_size).to('cuda')
        pipeline.init_weights()
    else:
        raise NotImplementedError(f"Regressor {config.regressor} not implemented")

    ## Fit model
    print(f'Training Regressor for {config.bmd_sub}. Input ROIs: {config.roi}. Target: {config.target}. Training data size: {len(dataset_train)}')
    if config.regressor == 'himalaya-ridge':
        pipeline.fit(dataset_train.X, dataset_train.y)
    else:
        dl_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
        dl_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)
        pipeline.fit_dl(dl_train, dl_test, epochs=40)
        # pipeline.fit(fmri_feat_train, target_train, X_test=fmri_feat_test, y_test=target_test)

    print("Evaluating and saving")
    save_path = f'estimated_vectors/{method}_sub{config.bmd_sub}_{config.target}'
    eval_and_save(save_path, pipeline, datasets=[dataset_test])

    # preds_train = pipeline.predict(dl_train)
    # preds_test = pipeline.predict(dl_test)

    # preds_train = to_numpy(preds_train)
    # preds_test = to_numpy(preds_test)
    # target_train = to_numpy(target_train)
    # target_test = to_numpy(target_test)

    # ## Save predictions
    # save_path = f'estimated_vectors/{method}_sub{config.bmd_sub}_{config.target}'
    # os.makedirs(save_path, exist_ok=True)
    # np.save(f'{save_path}/preds_train.npy', preds_train)
    # np.save(f'{save_path}/preds_test.npy', preds_test)

    # ## Compute test metrics
    # print("Train metrics:")
    # train_metrics = compute_metrics(target_train, preds_train, verbose=True)
    # print(train_metrics)

    # print("Test metrics:")
    # test_metrics = compute_metrics(target_test, preds_test, verbose=True)

    # ## Save metrics dict as pkl
    # with open(f'{save_path}/test_metrics:{test_metrics}.pkl', 'wb') as f:
    #     pkl.dump(test_metrics, f)


def eval_and_save(save_path, pipeline, datasets):
    os.makedirs(save_path, exist_ok=True)

    for d in datasets:
        d.return_filename = True
        pred_and_targ_dict = {}
        all_preds = []
        all_targets = []
        for x, y, _, target_filename in d:
            preds = pipeline.predict(x[None])
            preds = to_numpy(preds)

            if target_filename not in pred_and_targ_dict:
                pred_and_targ_dict[target_filename] = {'preds': [], 'targ': None}
            
            pred_and_targ_dict[target_filename]['preds'].append(preds)
            pred_and_targ_dict[target_filename]['targ'] = y[None]

        for target_filename, pt in pred_and_targ_dict.items():
            avg_preds = np.mean(pt['preds'], axis=0)
            all_preds.append(avg_preds)
            all_targets.append(pt['targ'])

            np.save(os.path.join(save_path, target_filename.split("/")[-1]), avg_preds)

        print("Metrics for", d.subset)
        metrics = compute_metrics(np.concatenate(all_targets), np.concatenate(all_preds), verbose=True)

        # ## Save metrics dict as pkl
        with open(f'{save_path}/_metrics:{metrics}.pkl', 'wb') as f:
            pkl.dump(metrics, f)

def to_numpy(arr):
    try:
        return maybe_move_to_host(arr).numpy()
    except AttributeError:
        pass
    return arr


def maybe_move_to_host(arr):
    """Moves array to host if it's on GPU"""
    try:
        return arr.detach().cpu()
    except AttributeError:
        pass
    return arr


def predict_and_average(pipeline, X_with_reps, n_reps=10):
    '''Makes predictions with different reps as input and averages the results'''

    preds = pipeline.predict(X_with_reps)

    preds = rearrange(preds, '(b r) n -> b r n', r=n_reps)
    preds = np.mean(preds, axis=1)

    return preds

def make_pipeline_for_himalaya_regressor(config, backend):
    # Define Ridge Regression Parameters
    if config.target in ['z_zeroscope', 'c_zeroscope']:
        # alphas = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
        alphas = [0.1, 1, 10]
    else:  # for larger number of outputs
        alphas = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alphas)

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )

    fmri_feat_train = backend.asarray(fmri_feat_train)
    fmri_feat_test = backend.asarray(fmri_feat_test)
    target_train = backend.asarray(target_train)
    target_test = backend.asarray(target_test)

    return pipeline


# def load_nsd_betas_impulse(
#     path_to_subject_data: str, roi: list, avg_train_reps=True
# ) -> None:
#     betas_impulse_train_list = []

#     for r in roi:
#         pkl_name = f'{r}_betas-GLMsingle_type-typeb_z=1.pkl'
#         with open(
#             os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb'
#         ) as f:
#             data = pkl.load(f)

#         if avg_train_reps:
#             betas_impulse_train_list.append(np.mean(data['data_allvoxel'], axis=1))
#         else:
#             # Concatenate all repetitions into dim 0
#             data_train = np.concatenate(
#                 [
#                     data['data_allvoxel'][:, i, :]
#                     for i in range(data['data_allvoxel'].shape[1])
#                 ]
#             )
#             betas_impulse_train_list.append(data_train)

#         # TODO: add noise ceiling

#     betas_impulse_train_npy = np.concatenate(betas_impulse_train_list, axis=1)

#     return betas_impulse_train_npy


# def load_boldmoments_betas_impulse(
#     path_to_subject_data: str, roi: list, avg_train_reps=True, concat_noise_ceiling=True
# ) -> None:
#     betas_impulse_train_list = []
#     betas_impulse_test_list = []

#     for r in roi:
#         pkl_name = f'{r}_betas-GLMsingle_type-typed_z=1.pkl'
#         with open(
#             os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb'
#         ) as f:
#             data = pkl.load(f)

#         if avg_train_reps:
#             betas_impulse_train_list.append(
#                 np.mean(data['train_data_allvoxel'], axis=1)
#             )
#         else:
#             # Concatenate all repetitions into dim 0
#             data_train = np.concatenate(
#                 [
#                     data['train_data_allvoxel'][:, i, :]
#                     for i in range(data['train_data_allvoxel'].shape[1])
#                 ]
#             )
#             betas_impulse_train_list.append(data_train)

#         betas_impulse_test_list.append(np.mean(data['test_data_allvoxel'], axis=1))

#         # TODO: add noise ceiling

#     betas_impulse_train_npy = np.concatenate(betas_impulse_train_list, axis=1)
#     betas_impulse_test_npy = np.concatenate(betas_impulse_test_list, axis=1)

#     return betas_impulse_train_npy, betas_impulse_test_npy


# def load_boldmoments_betas_raw(
#     path_to_subject_data: str, roi: list, avg_train_reps=True
# ) -> None:
#     """
#     load fMRI data into list that can be used for regression.
#     List should contain N elements, corresponding to the N videos in the selected subset.
#     Each element is an array of betas, concatenating betas for all rois in the roi list.
#     """

#     fmri_features_train_list = []
#     fmri_features_test_list = []
#     for r in roi:
#         pkl_name = f"{r}_TRavg-56789_testing.pkl"
#         with open(os.path.join(path_to_subject_data, pkl_name), 'rb') as f:
#             data = pkl.load(f)

#         # Average over repetitions
#         if avg_train_reps:
#             fmri_features_train_list.append(np.mean(data['train_data'], axis=1))
#         else:
#             fmri_features_train_list.append(data['train_data'])

#     fmri_features_train_npy = np.concatenate(fmri_features_train_list, axis=1)
#     fmri_features_test_npy = np.concatenate(fmri_features_test_list, axis=1)

#     return fmri_features_train_npy, fmri_features_test_npy


# def load_target_vectors_nsd(path_to_target_vectors: str, subject: str, repeat_train=1, num_frames_to_simulate=15) -> None:
#     """
#     Load target vectors for a given subject
#     """
#     target_train = []
#     target_test = []

#     # Load training pickle
#     with open(f'data/betas_nsd/{subject}/events_imgtag-73k_id.pkl', 'rb') as f:
#         img_idxs = pkl.load(f)

#     train_list = img_idxs[0]
#     for target in train_list:
#         vec = np.load(f'{path_to_target_vectors}/{target-1:06d}.npy')
#         # Repeat vector for each frame to simulate
#         target_train.append(np.repeat(vec[None], num_frames_to_simulate, axis=0))

#     target_train = np.array(target_train * repeat_train)

#     # flatten vectors
#     target_train = target_train.reshape(target_train.shape[0], -1)
#     print('target_train.shape after flatten', target_train.shape)

#     print(
#         f"Loaded {len(img_idxs[0])} NSD img_idxs for subject {subject}, starting with {img_idxs[0][0:5]}"
#     )

#     return target_train


# def load_target_vectors_boldmoments(
#     path_to_target_vectors: str, repeat_train=1
# ) -> None:
#     """
#     Load target vectors for a given subject
#     """
#     target_train = []
#     target_test = []

#     train_list = list(range(1001))

#     for target in os.listdir(path_to_target_vectors):
#         if int(target.split('.')[0]) in train_list:
#             target_train.append(np.load(f'{path_to_target_vectors}/{target}'))
#         else:
#             target_test.append(np.load(f'{path_to_target_vectors}/{target}'))

#     target_train = np.array(target_train * repeat_train)
#     target_test = np.array(target_test)

#     # flatten vectors
#     target_train = target_train.reshape(target_train.shape[0], -1)
#     target_test = target_test.reshape(target_test.shape[0], -1)

#     return target_train, target_test


if __name__ == "__main__":
    # Argument handling
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Target vector to regress. One of z_zeroscope, c_zeroscope, blip",
    )

    parser.add_argument(
        "--config",
        type=str,
        default='config/regression_config.yaml',
        help="Path to config file",
    )

    parser.add_argument(
        "--betas_type",
        type=str,
        default='betas_impulse',
        help="fMRI signals to use as features. One of betas_raw, betas_impulse",
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=['BMDgeneral'],
        nargs="*",
        help=f"ROIs to use as features. Use WB for whole brain, or a combination of ROIs, e.g. EBA PPA LOC. ROIs will be concatenated.",
    )

    parser.add_argument(
        "--bmd_sub",
        type=int,
        default=[1],
        nargs="*",
        help="List of BMD subjects for which fMRI data will be used. Ints from 1 to 10.",
    )

    parser.add_argument(
        "--nsd_sub",
        type=int,
        default=[1],
        nargs="*",
        help="List of NSD subjects for which fMRI data will be used. Ints from 1 to 8.",
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default='mlp',
        help="Regressor to use. One of himalaya-ridge, autogluon, mlp",
    )


    parser.add_argument(
        "--hidden_size",
        type=int,
        default=2048,
        help="Hidden size of MLP regressor",
    )

    parser.add_argument(
        "--avg_train_reps",
        type=bool,
        default=False,
        help="Whether to use individual reps or averaged ones during training",
    )

    parser.add_argument(
        "--use_nsd",
        type=bool,
        default=False,
        help="Whether to use NSD data as well",
    )

    args = parser.parse_args()
    main(args)
