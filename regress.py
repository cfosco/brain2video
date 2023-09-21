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

class HimalayaRidgeRegressor:
    def __init__(self, alphas):
        self.alphas = alphas
        self.regressor = RidgeCV(alphas)
        self.mean = None
        self.std = None

    def fit(self, X, y):
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
    
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 100)
        self.fc2 = nn.Linear(100, output_shape)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(100)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        # x = self.norm(x)
        x = self.fc2(x)
        return x

    def fit(self, X, y, batch_size=100, opt='adam', epochs=500, lr=0.001, verbose=True, use_tqdm=False):
        """Training function with a set Adam optimizer"""
        self.train()
        if opt == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise NotImplementedError("Optimizer not defined")
        criterion = nn.MSELoss().to('cuda')

        # Transfomr X and y into pytorch tensors and send to GPU
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float().to('cuda')
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float().to('cuda')       

        for epoch in range(epochs):
            if use_tqdm:
                pbar = tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch}")
            else:
                pbar = range(0, len(X), batch_size)
            for i in pbar:
                optimizer.zero_grad()
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                # print dtypes
                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                if use_tqdm:
                    pbar.set_postfix({'loss': loss.item()})
            if verbose:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch} Loss {loss.item()}')

    def predict(self, X):
        """Predict function"""
        with torch.no_grad():
            self.eval()
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X).float().to('cuda')
            preds = self(X).detach().cpu().numpy()
        return preds


class AutoGluonRegressor:
    pass


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Target vector to regress. One of z_zeroscope, c_zeroscope",
    )

    parser.add_argument(
        "--fmritype",
        type=str,
        default='betas',
        help="fMRI signals to use as features. One of betas, betas_significant",
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=['WB'],
        nargs="*",
        help=f"ROIs to use as features. Use WB for whole brain, or a combination of ROIs, e.g. EBA PPA LOC. ROIs will be concatenated.",
    )

    parser.add_argument(
        "--subject",
        type=str,
        default='sub01',
        help="Subject for which fMRI data will be used. One of sub01, sub02,... , sub10",
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default='mlp',
        help="Regressor to use. One of himalaya-ridge, autogluon, mlp",
    )

    args = parser.parse_args()
    target = args.target
    fmri_type = args.fmritype
    roi = args.roi
    subject = args.subject
    regressor = args.regressor
    backend = set_backend("torch_cuda")  # or "torch_cuda"

    ## Paths to fMRI data (our input features) and vectors to regress (our output targets)
    fmri_path = f'data/{fmri_type}/{subject}'
    targets_path = f'data/target_vectors/{target}'

    ## Build method string
    method = f'regressor:{regressor}_fmritype:{fmri_type}_rois:{"-".join(roi)}'

    ## Load train and test input features
    fmri_feat_train, fmri_feat_test = load_boldmoments_fmri(fmri_path, roi=roi)
    if regressor == 'himalaya-ridge':
        fmri_feat_train = backend.asarray(fmri_feat_train)
        fmri_feat_test = backend.asarray(fmri_feat_test)

    ## Load train and test output targets
    target_train, target_test = load_target_vectors(targets_path)
    if regressor == 'himalaya-ridge':
        target_train = backend.asarray(target_train)
        target_test = backend.asarray(target_test)



    ## Define regressor
    if regressor == 'himalaya-ridge':
        # Define Ridge Regression Parameters
        if target in ['z_zeroscope', 'c_zeroscope']:
            # alphas = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
            alphas = [0.1, 1, 10]
        else: # for much larger number of voxels
            alphas = [10000, 20000, 40000]

        ridge = RidgeCV(alphas=alphas)

        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
        )
        pipeline = make_pipeline(
            # preprocess_pipeline,
            ridge,
        )    
    elif regressor == 'mlp':
        input_shape = fmri_feat_train.shape[1]
        output_shape = target_train.shape[1]
        pipeline = MLPRegressor(input_shape, output_shape).to('cuda')
    else:
        raise NotImplementedError("Regressor not defined")


    ## Fit model
    print(f'Training Regressor for {subject}. Input ROIs: {roi}. Target: {target}')
    print(f'Input features train shape: {fmri_feat_train.shape}')
    print(f'Target train shape: {target_train.shape}')

    print("Fitting model")
    pipeline.fit(fmri_feat_train, target_train)
    preds_train = pipeline.predict(fmri_feat_train)
    preds_test = pipeline.predict(fmri_feat_test)
    
    ## Save predictions
    save_path = f'estimated_vectors/{method}_{subject}_{target}'
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/preds_train.npy', preds_train)
    np.save(f'{save_path}/preds_test.npy', preds_test)

    ## Compute test metrics
    metrics = compute_metrics(target_test, preds_test, verbose=True)

    ## Save metrics dict as pkl
    with open(f'{save_path}/metrics:{metrics}.pkl', 'wb') as f:
        pkl.dump(metrics, f)



def load_boldmoments_fmri(path_to_subject_data: str, roi: list) -> None:
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
        fmri_features_train_list.append( np.mean(data['train_data'], axis=1))
        fmri_features_test_list.append( np.mean(data['test_data'], axis=1))

    fmri_features_train_npy = np.concatenate(fmri_features_train_list, axis=1) 
    fmri_features_test_npy = np.concatenate(fmri_features_test_list, axis=1)

    return fmri_features_train_npy, fmri_features_test_npy



def load_target_vectors(path_to_target_vectors: str) -> None:
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

    target_train = np.array(target_train)
    target_test = np.array(target_test)
    return target_train, target_test



if __name__ == "__main__":
    main()
