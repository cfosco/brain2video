from torch import nn
import torch
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class NNMixIn:
    """MixIn that introduces fit() and predict() interfaces for Neural Networks"""

    def fit_dl(self, 
               train_dataloader, 
               test_dataloader=None, 
               opt='adam',
               lr=0.001, 
               l1_lambda=1e-8,
               l2_lambda=1e-4,
               epochs=50, 
               verbose=True, 
               use_tqdm=False):
        
        self.train()

        if opt == 'adam':
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=l2_lambda)
        else:
            raise NotImplementedError("Optimizer not defined")
        criterion = nn.MSELoss().to('cuda')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=2, verbose=verbose
        )

        for epoch in range(epochs):
            if use_tqdm:
                pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}")
            else:
                pbar = enumerate(train_dataloader)
            for i, (batch_X, batch_y) in pbar:
                opt.zero_grad()
                batch_X = batch_X.float().to('cuda')
                batch_y = batch_y.float().to('cuda')

                preds = self(batch_X)
                loss = criterion(preds, batch_y)


                # L1 regularization
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = loss + l1_lambda * l1_norm

                loss.backward()
                opt.step()
                if verbose:
                    if i % 100 == 0:
                        print(f'Epoch {epoch} Loss {loss.item()}')

            # Compute validation error
            if test_dataloader is not None:
                if verbose and epoch % 2 == 0:
                    with torch.no_grad():
                        self.eval()
                        total_loss = 0.0  # Initialize total loss
                        total_items = 0  # Initialize total number of items
                        for batch_X, batch_y in test_dataloader:
                            batch_X = batch_X.float().to('cuda')
                            batch_y = batch_y.float().to('cuda')
                            preds_test = self(batch_X)
                            loss = criterion(preds_test, batch_y)
                            total_loss += loss.item() * batch_X.size(0)  # Add current batch loss to total loss
                            total_items += batch_X.size(0)  # Add number of items in current batch to total items
                        avg_loss = total_loss / total_items  # Compute average loss per item
                        print(f'Epoch {epoch} Average Validation Loss per Item {avg_loss}')  # Print average loss per item

                        scheduler.step(avg_loss)
                
                    self.train()

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
        """Training function for full datasets loaded in X and y arrays. Does not use batches or dataloaders"""
        
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


    def predict_dl(self, dataloader):
        """Predict function"""
        preds=[]
        with torch.no_grad():
            self.eval()
            for b in dataloader:
                batch_X = b[0].float().to('cuda')
                preds.append(self(batch_X))           
        return torch.cat(preds, dim=0)


    # def predict(self, X):
    #     """Predict function"""
    #     with torch.no_grad():
    #         self.eval()
    #         for i in range(0, len(X), batch_size):
    #             batch_X = X[i : i + batch_size]
    #             batch_X = torch.tensor(batch_X).float().to('cuda')
    #             preds = self(batch_X)
    #             if i == 0:
    #                 preds_all = preds
    #             else:
    #                 preds_all = torch.cat([preds_all, preds], dim=0)
    #     return preds_all



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


class MLPRegressor(nn.Module, NNMixIn):
    """Simple MLP regressor with 1 hidden layer"""

    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 hidden_size=2000, 
                 drop=0.3):
        super().__init__()
        print("Initializing MLPRegressor with: ")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        print(f"Hidden size: {hidden_size}")
        self.fc1 = nn.Linear(input_shape, hidden_size, bias=True)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_shape, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        # Print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class SwiGLURegressor(nn.Module, NNMixIn):
    """ SwiGLU Regressor
    """
    def __init__(
            self,
            in_features,
            out_features=None,
            hidden_features=None,
            act_layer=nn.SiLU,
            norm_layer=nn.BatchNorm1d,
            bias=True,
            drop=0.3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

     
    # def fit_dl(self, 
    #            train_dataloader, 
    #            test_dataloader=None, 
    #            opt='adam',
    #            lr=0.001, 
    #            l1_lambda=1e-8,
    #            l2_lambda=1e-4,
    #            epochs=50, 
    #            verbose=True, 
    #            use_tqdm=False):
        
    #     self.train()

    #     if opt == 'adam':
    #         opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=l2_lambda)
    #     else:
    #         raise NotImplementedError("Optimizer not defined")
    #     criterion = nn.MSELoss().to('cuda')
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         opt, mode='min', factor=0.1, patience=2, verbose=verbose
    #     )

    #     for epoch in range(epochs):
    #         if use_tqdm:
    #             pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}")
    #         else:
    #             pbar = enumerate(train_dataloader)
    #         for i, (batch_X, batch_y) in pbar:
    #             opt.zero_grad()
    #             batch_X = batch_X.float().to('cuda')
    #             batch_y = batch_y.float().to('cuda')

    #             preds = self(batch_X)
    #             loss = criterion(preds, batch_y)


    #             # L1 regularization
    #             l1_norm = sum(p.abs().sum() for p in self.parameters())
    #             loss = loss + l1_lambda * l1_norm

    #             loss.backward()
    #             opt.step()
    #             if verbose:
    #                 if i % 100 == 0:
    #                     print(f'Epoch {epoch} Loss {loss.item()}')

    #         # Compute validation error
    #         if test_dataloader is not None:
    #             if verbose and epoch % 2 == 0:
    #                 with torch.no_grad():
    #                     self.eval()
    #                     total_loss = 0.0  # Initialize total loss
    #                     total_items = 0  # Initialize total number of items
    #                     for batch_X, batch_y in test_dataloader:
    #                         batch_X = batch_X.float().to('cuda')
    #                         batch_y = batch_y.float().to('cuda')
    #                         preds_test = self(batch_X)
    #                         loss = criterion(preds_test, batch_y)
    #                         total_loss += loss.item() * batch_X.size(0)  # Add current batch loss to total loss
    #                         total_items += batch_X.size(0)  # Add number of items in current batch to total items
    #                     avg_loss = total_loss / total_items  # Compute average loss per item
    #                     print(f'Epoch {epoch} Average Validation Loss per Item {avg_loss}')  # Print average loss per item

    #                     scheduler.step(avg_loss)
                
    #                 self.train()

    # def fit(
    #     self,
    #     X,
    #     y,
    #     X_test=None,
    #     y_test=None,
    #     batch_size=50,
    #     opt='adam',
    #     epochs=300,
    #     lr=0.001,
    #     verbose=True,
    #     use_tqdm=False,
    # ):
    #     """Training function with a set Adam optimizer"""
    #     self.train()
    #     if opt == 'adam':
    #         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
    #     else:
    #         raise NotImplementedError("Optimizer not defined")
    #     criterion = nn.MSELoss().to('cuda')

    #     # Transform X and y into pytorch tensors and send to GPU
    #     # if not isinstance(X, torch.Tensor):
    #     #     X = torch.tensor(X).float().to('cuda')
    #     # if not isinstance(y, torch.Tensor):
    #     #     y = torch.tensor(y).float().to('cuda')
    #     if X_test is not None and not isinstance(X_test, torch.Tensor):
    #         X_test = torch.tensor(X_test).float().to('cuda')
    #     if y_test is not None and not isinstance(y_test, torch.Tensor):
    #         y_test = torch.tensor(y_test).float().to('cuda')

    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.1, patience=10, verbose=verbose
    #     )

    #     for epoch in range(epochs):
    #         if use_tqdm:
    #             pbar = tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch}")
    #         else:
    #             pbar = range(0, len(X), batch_size)
    #         for i in pbar:
    #             optimizer.zero_grad()
    #             batch_X = X[i : i + batch_size]
    #             batch_y = y[i : i + batch_size]

    #             # send batch to GPU
    #             batch_X = torch.tensor(batch_X).float().to('cuda')
    #             batch_y = torch.tensor(batch_y).float().to('cuda')

    #             preds = self(batch_X)
    #             loss = criterion(preds, batch_y)
    #             loss.backward()
    #             optimizer.step()
    #             if use_tqdm:
    #                 pbar.set_postfix({'loss': loss.item()})
    #         if verbose:
    #             if epoch % 10 == 0:
    #                 print(f'Epoch {epoch} Loss {loss.item()}')

    #         # Compute validation error
    #         if X_test is not None and y_test is not None:
    #             if verbose and epoch % 10 == 0:
    #                 with torch.no_grad():
    #                     self.eval()
    #                     preds_test = self(X_test)
    #                     loss = criterion(preds_test, y_test)
    #                     print(f'Epoch {epoch} Validation Loss {loss.item()}')
    #                 self.train()

    #         scheduler.step(loss)


    # def predict_dl(self, dataset):
    #     """Predict function"""
    #     preds=[]
    #     with torch.no_grad():
    #         self.eval()
    #         for batch_X, batch_y in dataset:
    #             batch_X = batch_X.float().to('cuda')
    #             preds.append(self(batch_X))           
    #     return torch.cat(preds, dim=0)


    # def predict(self, X, batch_size=100):
    #     """Predict function"""
    #     with torch.no_grad():
    #         self.eval()
    #         for i in range(0, len(X), batch_size):
    #             batch_X = X[i : i + batch_size]
    #             batch_X = torch.tensor(batch_X).float().to('cuda')
    #             preds = self(batch_X)
    #             if i == 0:
    #                 preds_all = preds
    #             else:
    #                 preds_all = torch.cat([preds_all, preds], dim=0)
    #     return preds_all


class ResidualBlock(nn.Module):
    """Residual Block with two linear layers, batch normalization, and dropout"""
    def __init__(self, dim, dropout_p):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_p)

        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_p)

        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout2(out)

        out += identity
        return out



class MLPRegressorResidual(nn.Module, NNMixIn):
    """MLP Regressor with residual blocks"""
    def __init__(self, input_dim, output_dim, hidden_dim=2048, dropout_p=0.3):
        super(MLPRegressor, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_p),
            ResidualBlock(hidden_dim, dropout_p),
            ResidualBlock(hidden_dim, dropout_p)
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x




class TransformerRegressor(nn.Module, NNMixIn):
    def __init__(self, input_dim, output_dim, n_heads, dim_feedforward=2048, n_layers=6):
        super(TransformerRegressor, self).__init__()
        self.input_linear = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_linear = nn.Linear(output_dim, output_dim)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.transformer_encoder(src)
        output = self.output_linear(src)
        return output


class AutoGluonRegressor:
    pass

