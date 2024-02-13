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

from models import MLPRegressor, SwiGLURegressor
from utils import compute_metrics
from dataset import *
from omegaconf import OmegaConf
import datetime

def main(args):

    # Load the configuration file
    config = OmegaConf.load('configs/multi_dataset_training.yaml')

    # Merge with command-line arguments
    # cli_conf = OmegaConf.from_cli()

    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    print("CLI Conf:", args_dict)
    config = OmegaConf.merge(config, OmegaConf.create(args_dict))
    print("Merged:",OmegaConf.to_yaml(config))

    print(config.train_on)
    print(config.test_on)
    print(config.subs)
    print(config.regressor)

    # Build method string
    method = (
        f'trainon:{"-".join(config.train_on)}_subs:{"-".join(map(str,config.subs))}_regressor:{config.regressor}'
    )
    print("Method:", method)

    # Set job name as current date + method
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    job_name = f"{method}_launchedAt:{now}"

    print("job_name:", job_name)

    # Prepare training and testing datasets
    dataset_train, dataset_test_dict = make_datasets(
        config.train_on, config.test_on, config
    )

    # Define regressor
    if config.regressor == "himalaya-ridge":
        backend = set_backend("torch")
        pipeline = make_pipeline_for_himalaya_regressor(config, backend)
    elif config.regressor == "mlp":
        # input_shape = fmri_feat_train.shape[1]
        # output_shape = target_train.shape[1]
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = MLPRegressor(
            input_shape, output_shape, hidden_size=config.mlp.hidden_size
        ).to("cuda")
    elif config.regressor == "swiglu":
        input_shape = dataset_train[0][0].shape[0]
        output_shape = dataset_train[0][1].shape[0]
        pipeline = SwiGLURegressor(
            input_shape, output_shape, hidden_features=config.swiglu.hidden_size
        ).to("cuda")
        pipeline.init_weights()
    else:
        raise NotImplementedError(f"Regressor {config.regressor} not implemented")

    # Train model
    print(
        f"Training Regressor for subs {config.subs}. "
        f"Input ROIs: {config.rois}. Target: {config.target}. "
        f"Training data size: {len(dataset_train)}"
    )

    if config.regressor == "himalaya-ridge":
        pipeline.fit(dataset_train.X, dataset_train.y)
    else:
        dl_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
        for d in dataset_test_dict:
            dl_test = DataLoader(dataset_test_dict[d], batch_size=config.batch_size, shuffle=False)

        train_model(
            pipeline,
            dl_train,
            dl_test,
            optimizer="adamw",
            criterion=torch.nn.MSELoss(),
            lr=0.001,
            l1_lambda=1e-8,
            l2_lambda=1e-4,
            epochs=1,
            device="cuda",
        )
        # pipeline.fit_dl(dl_train, dl_test_dict, epochs=2)
        # pipeline.fit(fmri_feat_train, target_train, X_test=fmri_feat_test, y_test=target_test)

    print("Evaluating and saving")
    vectors_save_path = f"estimated_vectors/{config.target}/{job_name}"
    metrics = eval_and_save_estimated_vectors(vectors_save_path, pipeline, datasets=dataset_test_dict)

    # Save logs: config used, training progress, final output metrics
    logs_save_path = "./evaluation"
    save_metrics(logs_save_path, job_name, config, metrics)


def save_metrics(logs_save_path, job_name, config, metrics):
    
    # round all metrics to 3 decimals
    metrics = {k: round(v, 3) for k, v in metrics.items()}

    # Define name of file
    filename = f"{logs_save_path}/job:{job_name}_metrics:{metrics}.yaml"

    # transform to yaml
    metrics = OmegaConf.to_yaml({'metrics':metrics})
    final_config = OmegaConf.to_yaml({'config':config})

    with open(filename, "w") as f:
        f.write(final_config)
        f.write("\n")
        f.write(metrics)



def make_datasets(train_on, test_on, config):

    # make_nsd_dataset = lambda x: NSDBetasAndTargetsDataset(
    #     betas_path=config.nsd_betas_path,
    #     targets_path=os.path.join(config.nsd_targets_path, config.target),
    #     avg_reps=False,
    #     rois=config.roi,
    #     subs=config.nsd_sub,
    #     subset=x,
    #     load_all_in_ram=False,
    #     num_frames_to_simulate=1 if config.target == "blip" else 15,
    # )

    # make_bmd_dataset = lambda x: BMDBetasAndTargetsDataset(
    #     betas_path=config.bmd_betas_path,
    #     targets_path=os.path.join(config.bmd_targets_path, config.target),
    #     avg_reps=False,
    #     rois=config.roi,
    #     subs=config.bmd_sub,
    #     subset=x,
    #     load_all_in_ram=False,
    # )

    # make_had_dataset = lambda x: HADBetasAndTargetsDataset(
    #     betas_path=config.had_betas_path,
    #     targets_path=os.path.join(config.had_targets_path, config.target),
    #     metadata_path="./data/metadata_had",
    #     avg_reps=False,
    #     beta_type="impulse",
    #     rois=config.roi,  # should basically be: ["Group41"]
    #     subs=config.had_sub,
    #     subset=x,
    #     load_all_in_ram=False,
    #     use_noise_ceiling=True,
    #     return_filename=False,
    #     flatten_targets=True,
    # )

    def maker(dataset_name, subset):
        return BT_DATASET_MAP[dataset_name](
            betas_path=DATASET_PATHS[dataset_name]['betas'], 
            targets_path=DATASET_PATHS[dataset_name]['targets']+f'/{config.target}',
            metadata_path=DATASET_PATHS[dataset_name]['metadata'],
            bundle_reps=config.bundle_reps,
            avg_reps=config.avg_train_reps,
            rois=config.rois,
            subs=config.subs,
            subset=subset,
            load_all_in_ram=False,
            use_noise_ceiling=False,
            return_filename=False,
            flatten_targets=True,
            )

    train_datasets = []
    test_datasets = {}



    for dataset_name in train_on:
        if dataset_name not in test_on:
            train_datasets.append(maker(dataset_name, subset="all"))
        else:
            train_datasets.append(maker(dataset_name, subset="train"))

    train_dataset = ConcatDataset(train_datasets)

    for dataset_name in test_on:
        test_datasets[dataset_name] = maker(dataset_name, subset="test")

    return train_dataset, test_datasets


def eval_and_save_estimated_vectors(save_path, pipeline, datasets, use_dataloader=True):
    """Evaluates over all datasets and saves predictions

    Args:
    save_path (str): path to save predictions
    pipeline : pipeline to use for predictions
    datasets (dict of datasets): datasets to evaluate. Typically train and test datasets from the same data.
      Each dataset should have a .subset attribute
    """
    os.makedirs(save_path, exist_ok=True)

    pipeline.eval()

    for name, d in datasets.items():
        print("Evaluating for dataset", name, "with", len(d), "samples, subset", d.subset)
        d.flatten_targets = False
        d.return_filename = True
        subset = d.subset

        if use_dataloader:
            d = DataLoader(d, batch_size=512, shuffle=False)

        pred_and_targ_dict = {}
        all_preds = []
        all_targets = []

        for betas_batch, targets_batch, _, targets_filenames_batch in d:
            if betas_batch.ndim == 2:
                betas_batch = betas_batch.unsqueeze(0)
                targets_batch = targets_batch.unsqueeze(0)
                targets_filenames_batch = [targets_filenames_batch]
            with torch.no_grad():
                betas_batch = betas_batch.float().to("cuda")
                targets_batch = targets_batch.float().to("cuda")
                preds = pipeline(betas_batch)
                preds = to_numpy(preds)
            # print("preds.shape, y.shape", preds.shape, y.shape)
                
            
            for preds, targets_batch, target_filename in zip(preds, targets_batch, targets_filenames_batch):
                if target_filename not in pred_and_targ_dict:
                    pred_and_targ_dict[target_filename] = {"preds": [], "targ": None}

                pred_and_targ_dict[target_filename]["preds"].append(preds)
                pred_and_targ_dict[target_filename]["targ"] = targets_batch

        # TODO: revamp this for loop: compute_metrics should return list of metrics for each element in the batch, then averge should be done when all batches have been processed
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

        print("Metrics for", subset)
        metrics = compute_metrics(
            np.concatenate(all_targets), np.concatenate(all_preds), verbose=True
        )

        # Save metrics dict as pkl
        with open(f"{save_path}/_{subset}_metrics:{metrics}.pkl", "wb") as f:
            pkl.dump(metrics, f)

    return metrics



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



if __name__ == "__main__":

    # Argument handling
    parser = argparse.ArgumentParser()


    # command line arguments will overwrite the default config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_dataset_training.yaml",
        help="Path to config file",
    ) 

    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Target vector to regress. One of z_zeroscope, c_zeroscope, blip, viclip",
    )


    parser.add_argument(
        "--train_on",
        type=str,
        default=["bmd", "had", "cc2017"],
        nargs="*",
        help="List of datasets to train on. One or multiple of bmd, had, cc2017",
    )

    parser.add_argument(
        "--test_on",
        type=str,
        default=["bmd", "had", "cc2017"],
        nargs="*",
        help="List of datasets to test on. One or multiple of bmd, had, cc2017",
    )

    parser.add_argument(
        "--rois",
        type=str,
        default=["Group41"],
        nargs="*",
        help="ROIs to use as features. Use Group41 to use our custom grouped vertices encompassing relevant voxels over the whole brain.",
    )

    parser.add_argument(    
        "--subs",
        type=int,
        nargs="*",
        default=[1],
        help="Subjects to use for training and testing.",
    )

    parser.add_argument(
        "--avg_train_reps",
        type=bool,
        default=False,
        help="Whether to use individual reps or averaged ones during training",
    )

    parser.add_argument(
        "--bundle_reps",
        type=bool,
        default=False,
        help="Whether to bundle repetitions or not",
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default="mlp",
        help="Regressor to use. One of mlp, swiglu, autogluon",
    )

    # parser.add_argument(
    #     "--hidden_size",
    #     type=int,
    #     default=2048,
    #     help="Hidden size of MLP regressor",
    # )

    args = parser.parse_args()
    main(args)
