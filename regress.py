import argparse
import contextlib
import os
import pickle as pkl

import numpy as np
import torch
from einops import rearrange
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from dataset import (
    BMDBetasAndTargetsDataset,
    HADBetasAndTargetsDataset,
    NSDBetasAndTargetsDataset,
)
from models import MLPRegressor, SwiGLURegressor
from utils import compute_metrics


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

    repeat_train = 1 if config.avg_train_reps else 3

    # Build method string
    method = (
        f'regressor:{config.regressor}withscheduleronval-hidden:{config.hidden_size}-rois:'
        f'{"-".join(config.roi)}-avgtrainreps:{config.avg_train_reps}-traindata:{config.train_on}'
    )
    print("Method:", method)

    # Prepare training and testing datasets
    dataset_train, dataset_test_dict = make_datasets(
        config.train_on, config.test_on, config
    )

    # # If pretrain NSD, load NSD data
    # if args.use_nsd:

    #     nsd_dataset_both = NSDBetasAndTargetsDataset(
    #                     betas_path=nsd_betas_path,
    #                     targets_path=nsd_targets_path,
    #                     avg_reps=False,
    #                     rois=config.roi,
    #                     subs=config.nsd_sub,
    #                     subset='both',
    #                     load_all_in_ram=False,
    #                     num_frames_to_simulate=1 if config.target=='blip' else 15)

    #     nsd_dataset_train = NSDBetasAndTargetsDataset(
    #                     betas_path=nsd_betas_path,
    #                     targets_path=nsd_targets_path,
    #                     avg_reps=False,
    #                     rois=config.roi,
    #                     subs=config.nsd_sub,
    #                     subset='train',
    #                     load_all_in_ram=False,
    #                     num_frames_to_simulate=1 if config.target=='blip' else 15)

    #     nsd_dataset_test = NSDBetasAndTargetsDataset(
    #                     betas_path=nsd_betas_path,
    #                     targets_path=nsd_targets_path,
    #                     avg_reps=False,
    #                     rois=config.roi,
    #                     subs=config.nsd_sub,
    #                     subset='test',
    #                     load_all_in_ram=False,
    #                     num_frames_to_simulate=1 if config.target=='blip' else 15)

    # fmri_feat_train_nsd = load_nsd_betas_impulse(
    #     f'data/betas_nsd/{subject}', roi=roi, avg_train_reps=config.avg_train_reps
    # )
    # target_train_nsd = load_target_vectors_nsd(
    #     f'data/target_vectors_nsd/{target}',
    #     subject=subject,  # NSD has different targets per subject, as not all subjects saw all stimuli
    #     repeat_train=repeat_train,
    # )

    # if args.use_bmd:

    # Load train and test input features
    # if fmri_type == 'betas_impulse':
    #     fmri_feat_train, fmri_feat_test = load_boldmoments_betas_impulse(
    #         fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
    #     )
    # elif fmri_type == 'betas_raw':
    #     fmri_feat_train, fmri_feat_test = load_boldmoments_betas_raw(
    #         fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
    #     )

    # Load train and test output targets
    # target_train, target_test = load_target_vectors_boldmoments(
    #     targets_path, repeat_train=repeat_train
    # )

    # Concatenate NSD data
    # if args.use_nsd:

    #     dataset_train = ConcatDataset([nsd_dataset_both, bmd_dataset_train])
    #     dataset_train.subset = 'train'
    #     dataset_train.return_filename = False
    #     print("Instantiated Train dataset as concatenation of NSD and BMD datasets")
    #     print("Length of nsd_dataset_both:", len(nsd_dataset_both))
    #     print("Length of bmd_dataset_train:", len(bmd_dataset_train))
    #     print("Length of concatenated dataset:", len(dataset_train))
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
    # else:
    #     dataset_train = bmd_dataset_train

    # dataset_test = bmd_dataset_test
    # print("Instantiated Test dataset as BMD dataset")
    # print("Length of bmd_dataset_test:", len(bmd_dataset_test))

    # Define regressor
    if config.regressor == "himalaya-ridge":
        pipeline = make_pipeline_for_himalaya_regressor(config, backend)
    elif config.regressor == "mlp":
        # input_shape = fmri_feat_train.shape[1]
        # output_shape = target_train.shape[1]
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = MLPRegressor(
            input_shape, output_shape, hidden_size=config.hidden_size
        ).to("cuda")
    elif config.regressor == "swiglu":
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = SwiGLURegressor(
            input_shape, output_shape, hidden_features=config.hidden_size
        ).to("cuda")
        pipeline.init_weights()
    else:
        raise NotImplementedError(f"Regressor {config.regressor} not implemented")

    # Train model
    print(
        f"Training Regressor for {config.bmd_sub}. "
        f"Input ROIs: {config.roi}. Target: {config.target}. "
        f"Training data size: {len(dataset_train)}"
    )

    if config.regressor == "himalaya-ridge":
        pipeline.fit(dataset_train.X, dataset_train.y)
    else:
        dl_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
        for d in dataset_test_dict:
            dl_test = DataLoader(dataset_test_dict[d], batch_size=512, shuffle=False)

        train_model(
            pipeline,
            dl_train,
            dl_test,
            optimizer="adamw",
            criterion=torch.nn.MSELoss(),
            lr=0.001,
            l1_lambda=1e-8,
            l2_lambda=1e-4,
            epochs=100,
            device="cuda",
        )
        # pipeline.fit_dl(dl_train, dl_test_dict, epochs=2)
        # pipeline.fit(fmri_feat_train, target_train, X_test=fmri_feat_test, y_test=target_test)

    print("Evaluating and saving")
    save_path = f"estimated_vectors/{config.target}/{method}-sub{config.bmd_sub}"
    eval_and_save(save_path, pipeline, datasets=dataset_test_dict)

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


def make_datasets(train_on, test_on, config):
    make_nsd_dataset = lambda x: NSDBetasAndTargetsDataset(
        betas_path=config.nsd_betas_path,
        targets_path=os.path.join(config.nsd_targets_path, config.target),
        avg_reps=False,
        rois=config.roi,
        subs=config.nsd_sub,
        subset=x,
        load_all_in_ram=False,
        num_frames_to_simulate=1 if config.target == "blip" else 15,
    )

    make_bmd_dataset = lambda x: BMDBetasAndTargetsDataset(
        betas_path=config.bmd_betas_path,
        targets_path=os.path.join(config.bmd_targets_path, config.target),
        avg_reps=False,
        rois=config.roi,
        subs=config.bmd_sub,
        subset=x,
        load_all_in_ram=False,
    )

    make_had_dataset = lambda x: HADBetasAndTargetsDataset(
        betas_path=config.had_betas_path,
        targets_path=os.path.join(config.had_targets_path, config.target),
        metadata_path="./data/metadata_had",
        avg_reps=False,
        beta_type="impulse",
        rois=config.roi,  # should basically be: ["Group41"]
        subs=config.had_sub,
        subset=x,
        load_all_in_ram=False,
        use_noise_ceiling=True,
        return_filename=False,
        flatten_targets=True,
    )

    dataset_makers = {
        "bmd": make_bmd_dataset,
        "had": make_had_dataset,
        "nsd": make_nsd_dataset,
    }

    train_datasets = []
    test_datasets = {}

    for dataset_name in train_on:
        maker = dataset_makers[dataset_name]
        if dataset_name not in test_on:
            train_datasets.append(maker("both"))
        else:
            train_datasets.append(maker("train"))

    train_dataset = ConcatDataset(train_datasets)

    for dataset_name in test_on:
        maker = dataset_makers[dataset_name]
        test_datasets[dataset_name] = maker("test")

    return train_dataset, test_datasets


def eval_and_save(save_path, pipeline, datasets):
    """Evaluates over all datasets and saves predictions and metrics

    Args:
    save_path (str): path to save predictions and metrics
    pipeline : pipeline to use for predictions
    datasets (dict of datasets): datasets to evaluate. Typically train and test datasets from the same data.
      Each dataset should have a .subset attribute
    """
    os.makedirs(save_path, exist_ok=True)

    for name, d in datasets.items():
        d.flatten_targets = False
        d.return_filename = True
        pred_and_targ_dict = {}
        all_preds = []
        all_targets = []
        for x, y, _, target_filename in d:
            preds = pipeline.predict(x[None])
            preds = to_numpy(preds)
            print("preds.shape, y.shape", preds.shape, y.shape)

            if target_filename not in pred_and_targ_dict:
                pred_and_targ_dict[target_filename] = {"preds": [], "targ": None}

            pred_and_targ_dict[target_filename]["preds"].append(preds)
            pred_and_targ_dict[target_filename]["targ"] = y

        for target_filename, pt in pred_and_targ_dict.items():
            avg_preds = np.mean(
                pt["preds"], axis=0
            )  # Test time augmentation on the repetitions
            all_preds.append(avg_preds)
            all_targets.append(
                pt["targ"].reshape(-1)[None]
            )  # Flatten targets to compute metrics
            # (TODO: check that this reshape is correctly matching the flattening done in the dataset)

            # Save predicted vectors in their original shape
            avg_preds_unflattened = avg_preds.reshape(
                pt["targ"].shape
            )  # TODO: Check that this reshaping reshapes in the right way
            np.save(
                os.path.join(save_path, target_filename.split("/")[-1]),
                avg_preds_unflattened,
            )

        print("Metrics for", d.subset)
        metrics = compute_metrics(
            np.concatenate(all_targets), np.concatenate(all_preds), verbose=True
        )

        # Save metrics dict as pkl
        with open(f"{save_path}/_{d.subset}_metrics:{metrics}.pkl", "wb") as f:
            pkl.dump(metrics, f)


def to_numpy(arr):
    with contextlib.suppress(AttributeError):
        return maybe_move_to_host(arr).numpy()
    return arr


def maybe_move_to_host(arr):
    """Moves array to host if it's on GPU"""
    with contextlib.suppress(AttributeError):
        return arr.detach().cpu()
    return arr


def predict_and_average(pipeline, X_with_reps, n_reps=10):
    """Makes predictions with different reps as input and averages the results"""

    preds = pipeline.predict(X_with_reps)

    preds = rearrange(preds, "(b r) n -> b r n", r=n_reps)
    preds = np.mean(preds, axis=1)

    return preds


def make_pipeline_for_himalaya_regressor(config, backend):
    # Define Ridge Regression Parameters
    if config.target in ["z_zeroscope", "c_zeroscope"]:
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


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    optimizer="adamw",
    criterion=torch.nn.MSELoss(),
    lr=0.001,
    l1_lambda=1e-8,
    l2_lambda=1e-4,
    epochs=100,
    device="cuda",
):
    model.train()

    if optimizer == "adam" or optimizer == "adamw" or optimizer is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_lambda)

    for epoch in range(epochs):
        for i, (batch_X, batch_y) in enumerate(train_dataloader):
            opt.zero_grad()
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float().to(device)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            opt.step()

        # Compute validation error
        if test_dataloader is not None:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                total_items = 0
                for batch_X, batch_y in test_dataloader:
                    batch_X = batch_X.float().to(device)
                    batch_y = batch_y.float().to(device)
                    preds_test = model(batch_X)
                    loss = criterion(preds_test, batch_y)
                    total_loss += loss.item() * batch_X.size(0)
                    total_items += batch_X.size(0)
                avg_loss = total_loss / total_items
                print(f"Epoch {epoch} Average Validation Loss per Item {avg_loss}")
            model.train()


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
        default="config/regression_config.yaml",
        help="Path to config file",
    )

    # parser.add_argument(
    #     "--betas_type",
    #     type=str,
    #     default='betas_impulse',
    #     help="fMRI signals to use as features. One of betas_raw, betas_impulse",
    # )

    parser.add_argument(
        "--roi",
        type=str,
        default=["BMDgeneral"],
        nargs="*",
        help="ROIs to use as features. Use WB for whole brain, or a combination of ROIs, e.g. EBA PPA LOC. ROIs will be concatenated.",
    )

    parser.add_argument(
        "--bmd_sub",
        type=int,
        default=[1],
        nargs="*",
        help="List of BMD subjects for which fMRI data will be used. Ints from 1 to 10.",
    )

    parser.add_argument(
        "--had_sub",
        type=int,
        default=[1],
        nargs="*",
        help="List of HAD subjects for which fMRI data will be used. Ints from 1 to 10.",
    )

    parser.add_argument(
        "--nsd_sub",
        type=int,
        default=[1],
        nargs="*",
        help="List of NSD subjects for which fMRI data will be used. Ints from 1 to 8.",
    )

    parser.add_argument(
        "--nsd_betas_path",
        type=str,
        default="data/betas_nsd",
        help="Path to NSD betas",
    )

    parser.add_argument(
        "--nsd_targets_path",
        type=str,
        default="data/target_vectors_nsd",
        help="Path to NSD target vectors",
    )

    parser.add_argument(
        "--bmd_betas_path",
        type=str,
        default="data/betas_impulse_bmd",
        help="Path to BMD betas",
    )

    parser.add_argument(
        "--bmd_targets_path",
        type=str,
        default="data/target_vectors_bmd",
        help="Path to BMD target vectors",
    )

    parser.add_argument(
        "--had_betas_path",
        type=str,
        default="data/betas_impulse_had",
        help="Path to had betas",
    )

    parser.add_argument(
        "--had_targets_path",
        type=str,
        default="data/target_vectors_had",
        help="Path to had target vectors",
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default="mlp",
        help="Regressor to use. One of mlp, swiglu, autogluon",
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

    parser.add_argument(
        "--train_on",
        type=str,
        default=["bmd", "nsd"],
        nargs="*",
        help="List of datasets to train on. One or multiple of bmd, nsd, nad",
    )

    parser.add_argument(
        "--test_on",
        type=str,
        default=["bmd", "nsd"],
        nargs="*",
        help="List of datasets to test on. One or multiple of bmd, nsd, nad",
    )

    args = parser.parse_args()
    main(args)
