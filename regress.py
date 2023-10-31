import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils import compute_metrics

from torch import nn
import torch
from tqdm import tqdm
import yaml
import wandb


class HimalayaRidgeRegressor:
    def __init__(self, alphas):
        self.alphas = alphas
        self.regressor = RidgeCV(alphas)
        self.mean = None
        self.std = None

    def fit(self, X, y, X_test=None, y_test=None):
        # Standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        X = (X - self.mean) / self.std

        self.regressor.fit(X, y)

    def predict(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Regressor not trained")

        # Standardize features
        X = (X - self.mean) / self.std

        return self.regressor.predict(X)


class MLPRegressor(nn.Module):
    """Simple MLP regressor with 1 hidden layer"""

    def __init__(self, input_shape, output_shape, hidden_size=1000):
        super().__init__()
        print("Initializing MLPRegressor with: ")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        print(f"Hidden size: {hidden_size}")
        self.fc1 = nn.Linear(
            input_shape,
            hidden_size,
        )
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_shape)
        self.act = nn.GELU()

        # Print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

    def fit(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        batch_size=50,
        opt='adam',
        epochs=300,
        lr=0.001,
        verbose=True,
        use_tqdm=False,
    ):
        """Training function with a set Adam optimizer"""
        self.train()
        if opt == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        else:
            raise NotImplementedError("Optimizer not defined")
        criterion = nn.MSELoss().to('cuda')

        # Transform X and y into pytorch tensors and send to GPU
        # if not isinstance(X, torch.Tensor):
        #     X = torch.tensor(X).float().to('cuda')
        # if not isinstance(y, torch.Tensor):
        #     y = torch.tensor(y).float().to('cuda')
        if X_test is not None and not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test).float().to('cuda')
        if y_test is not None and not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test).float().to('cuda')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=verbose
        )

        for epoch in range(epochs):
            if use_tqdm:
                pbar = tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch}")
            else:
                pbar = range(0, len(X), batch_size)
            for i in pbar:
                optimizer.zero_grad()
                batch_X = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]

                # send batch to GPU
                batch_X = torch.tensor(batch_X).float().to('cuda')
                batch_y = torch.tensor(batch_y).float().to('cuda')

                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                if use_tqdm:
                    pbar.set_postfix({'loss': loss.item()})
            if verbose:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch} Loss {loss.item()}')

            # Compute validation error
            if X_test is not None and y_test is not None:
                if verbose and epoch % 10 == 0:
                    with torch.no_grad():
                        self.eval()
                        preds_test = self(X_test)
                        loss = criterion(preds_test, y_test)
                        print(f'Epoch {epoch} Validation Loss {loss.item()}')
                    self.train()

            scheduler.step(loss)

    def predict(self, X, batch_size=100):
        """Predict function"""
        with torch.no_grad():
            self.eval()
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]
                batch_X = torch.tensor(batch_X).float().to('cuda')
                preds = self(batch_X)
                if i == 0:
                    preds_all = preds
                else:
                    preds_all = torch.cat([preds_all, preds], dim=0)
        return preds_all


class AutoGluonRegressor:
    pass


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

    target = config.target
    fmri_type = config.fmritype
    roi = config.roi
    subject = config.subject[0]
    regressor = config.regressor
    backend = set_backend("numpy")  # or "torch_cuda"

    if config.avg_train_reps:
        repeat_train = 1
    else:
        repeat_train = 3

    ## Paths to fMRI data (our input features) and vectors to regress (our output targets)
    fmri_path = f'data/{fmri_type}/{subject}'
    targets_path = f'data/target_vectors/{target}_unflattened'

    ## Build method string
    method = f'regressor:{regressor}withscheduler_fmritype:{fmri_type}_rois:{"-".join(roi)}_avgtrainreps:{config.avg_train_reps}_usensd:{config.use_nsd}'
    print("Method:", method)

    # If pretrain NSD, load NSD data
    if args.use_nsd:
        fmri_feat_train_nsd = load_nsd_betas_impulse(
            f'data/betas_nsd/{subject}', roi=roi, avg_train_reps=config.avg_train_reps
        )
        target_train_nsd = load_target_vectors_nsd(
            f'data/target_vectors_nsd/{target}',
            subject=subject,  # NSD has different targets per subject, as not all subjects saw all stimuli
            repeat_train=repeat_train,
        )

    ## Load train and test input features
    if fmri_type == 'betas_impulse':
        fmri_feat_train, fmri_feat_test = load_boldmoments_betas_impulse(
            fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
        )
    elif fmri_type == 'betas_raw':
        fmri_feat_train, fmri_feat_test = load_boldmoments_betas_raw(
            fmri_path, roi=roi, avg_train_reps=config.avg_train_reps
        )

    ## Load train and test output targets
    target_train, target_test = load_target_vectors_boldmoments(
        targets_path, repeat_train=repeat_train
    )

    # Concatenate NSD data
    if args.use_nsd:
        print("Shapes before concatenating NSD data:")
        print(f"fmri_feat_train: {fmri_feat_train.shape}")
        print(f"target_train: {target_train.shape}")
        print(f"fmri_feat_train_nsd: {fmri_feat_train_nsd.shape}")
        print(f"target_train_nsd: {target_train_nsd.shape}")
        fmri_feat_train = np.concatenate([fmri_feat_train, fmri_feat_train_nsd], axis=0)
        target_train = np.concatenate([target_train, target_train_nsd], axis=0)

        print("Shapes after concatenating NSD data:")
        print(f"fmri_feat_train: {fmri_feat_train.shape}")
        print(f"target_train: {target_train.shape}")

    ## Define regressor
    if regressor == 'himalaya-ridge':
        # Define Ridge Regression Parameters
        if target in ['z_zeroscope', 'c_zeroscope']:
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

    elif regressor == 'mlp':
        input_shape = fmri_feat_train.shape[1]
        output_shape = target_train.shape[1]
        pipeline = MLPRegressor(input_shape, output_shape).to('cuda')
    else:
        raise NotImplementedError(f"Regressor {regressor} not implemented")

    ## Fit model
    print(f'Training Regressor for {subject}. Input ROIs: {roi}. Target: {target}')
    print(f'Input features train shape: {fmri_feat_train.shape}')
    print(f'Target train shape: {target_train.shape}')

    print("Fitting model")
    if regressor == 'himalaya-ridge':
        pipeline.fit(fmri_feat_train, target_train)
    else:
        pipeline.fit(
            fmri_feat_train, target_train, X_test=fmri_feat_test, y_test=target_test
        )
    preds_train = pipeline.predict(fmri_feat_train)
    preds_test = pipeline.predict(fmri_feat_test)

    preds_train = to_numpy(preds_train)
    preds_test = to_numpy(preds_test)
    target_train = to_numpy(target_train)
    target_test = to_numpy(target_test)

    ## Save predictions
    save_path = f'estimated_vectors/{method}_{subject}_{target}'
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/preds_train.npy', preds_train)
    np.save(f'{save_path}/preds_test.npy', preds_test)

    ## Compute test metrics
    print("Train metrics:")
    train_metrics = compute_metrics(target_train, preds_train, verbose=True)
    print(train_metrics)

    print("Test metrics:")
    test_metrics = compute_metrics(target_test, preds_test, verbose=True)

    ## Save metrics dict as pkl
    with open(f'{save_path}/test_metrics:{test_metrics}.pkl', 'wb') as f:
        pkl.dump(test_metrics, f)


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

    from einops import rearrange

    preds = pipeline.predict(X_with_reps)

    preds = rearrange(preds, '(b r) n -> b r n', r=n_reps)
    preds = np.mean(preds, axis=1)

    return preds


def load_nsd_betas_impulse(
    path_to_subject_data: str, roi: list, avg_train_reps=True
) -> None:
    betas_impulse_train_list = []

    for r in roi:
        pkl_name = f'{r}_betas-GLMsingle_type-typeb_z=1.pkl'
        with open(
            os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb'
        ) as f:
            data = pkl.load(f)

        if avg_train_reps:
            betas_impulse_train_list.append(np.mean(data['data_allvoxel'], axis=1))
        else:
            # Concatenate all repetitions into dim 0
            data_train = np.concatenate(
                [
                    data['data_allvoxel'][:, i, :]
                    for i in range(data['data_allvoxel'].shape[1])
                ]
            )
            betas_impulse_train_list.append(data_train)

        # TODO: add noise ceiling

    betas_impulse_train_npy = np.concatenate(betas_impulse_train_list, axis=1)

    return betas_impulse_train_npy


def load_boldmoments_betas_impulse(
    path_to_subject_data: str, roi: list, avg_train_reps=True, concat_noise_ceiling=True
) -> None:
    betas_impulse_train_list = []
    betas_impulse_test_list = []

    for r in roi:
        pkl_name = f'{r}_betas-GLMsingle_type-typed_z=1.pkl'
        with open(
            os.path.join(path_to_subject_data, 'prepared_allvoxel_pkl', pkl_name), 'rb'
        ) as f:
            data = pkl.load(f)

        if avg_train_reps:
            betas_impulse_train_list.append(
                np.mean(data['train_data_allvoxel'], axis=1)
            )
        else:
            # Concatenate all repetitions into dim 0
            data_train = np.concatenate(
                [
                    data['train_data_allvoxel'][:, i, :]
                    for i in range(data['train_data_allvoxel'].shape[1])
                ]
            )
            betas_impulse_train_list.append(data_train)

        betas_impulse_test_list.append(np.mean(data['test_data_allvoxel'], axis=1))

        # TODO: add noise ceiling

    betas_impulse_train_npy = np.concatenate(betas_impulse_train_list, axis=1)
    betas_impulse_test_npy = np.concatenate(betas_impulse_test_list, axis=1)

    return betas_impulse_train_npy, betas_impulse_test_npy


def load_boldmoments_betas_raw(
    path_to_subject_data: str, roi: list, avg_train_reps=True
) -> None:
    """
    load fMRI data into list that can be used for regression.
    List should contain N elements, corresponding to the N videos in the selected subset.
    Each element is an array of betas, concatenating betas for all rois in the roi list.
    """

    fmri_features_train_list = []
    fmri_features_test_list = []
    for r in roi:
        pkl_name = f"{r}_TRavg-56789_testing.pkl"
        with open(os.path.join(path_to_subject_data, pkl_name), 'rb') as f:
            data = pkl.load(f)

        # Average over repetitions
        if avg_train_reps:
            fmri_features_train_list.append(np.mean(data['train_data'], axis=1))
        else:
            fmri_features_train_list.append(data['train_data'])

    fmri_features_train_npy = np.concatenate(fmri_features_train_list, axis=1)
    fmri_features_test_npy = np.concatenate(fmri_features_test_list, axis=1)

    return fmri_features_train_npy, fmri_features_test_npy


def load_target_vectors_nsd(
    path_to_target_vectors: str, subject: str, repeat_train=1
) -> None:
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

    target_train = np.array(target_train * repeat_train)

    # flatten vectors
    target_train = target_train.reshape(target_train.shape[0], -1)

    print(
        f"Loaded {len(img_idxs[0])} NSD img_idxs for subject {subject}, starting with {img_idxs[0][0:5]}"
    )

    return target_train


def load_target_vectors_boldmoments(
    path_to_target_vectors: str, repeat_train=1
) -> None:
    """
    Load target vectors for a given subject
    """
    target_train = []
    target_test = []

    train_list = list(range(1001))

    for target in os.listdir(path_to_target_vectors):
        if int(target.split('.')[0]) in train_list:
            target_train.append(np.load(f'{path_to_target_vectors}/{target}'))
        else:
            target_test.append(np.load(f'{path_to_target_vectors}/{target}'))

    target_train = np.array(target_train * repeat_train)
    target_test = np.array(target_test)

    # flatten vectors
    target_train = target_train.reshape(target_train.shape[0], -1)
    target_test = target_test.reshape(target_test.shape[0], -1)

    return target_train, target_test


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
        "--fmritype",
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
        "--subject",
        type=str,
        default=['sub01'],
        nargs="*",
        help="Subject for which fMRI data will be used. One of sub01, sub02,... , sub10",
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default='mlp',
        help="Regressor to use. One of himalaya-ridge, autogluon, mlp",
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
