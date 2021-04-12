from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
import torch

from nemo.model.layers import Identity


class ElasticNet(LightningModule):
    def __init__(self, config):
        ''' Elastic Net built in PyTorch '''
        
        super().__init__()
        
        self._set_hparams(config)

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError('alpha should be >= 0.0 and <= 1.0')

        if self.weight_samples:
            if self.loss_fn.reduction != 'none':
                raise ValueError('loss_fn should have reduction=none if weight_samples is True.')
        
        self.strf = torch.nn.Linear(
            self.in_h * self.in_w * self.n_frames, 
            self.n_neurons
        )

        self.callbacks = [
            EarlyStopping(
                monitor = 'val_loss',
                patience = self.patience,
                min_delta = self.tol,
                mode = 'min'
            )
        ]


    def _set_hparams(self, config):
        ''' Sets default args if param not in config '''

        try:
            self.n_neurons = config['n_neurons']
            self.in_h = config['in_h']
            self.in_w = config['in_w']
            self.n_frames = config['n_frames']
        except KeyError:
            print('n_neurons, in_h, in_w, and n_frames are required arguments.')
            raise

        self.alpha = config['alpha'] if 'alpha' in config.keys() else 0.5
        self.lr = config['lr'] if 'lr' in config.keys() else 1e-4
        self.lambd = config['lambd'] if 'lambd' in config.keys() else 1e-4
        self.optim = config['optim'] if 'optim' in config.keys() else torch.optim.Adam
        self.loss_fn = config['loss_fn'] if 'loss_fn' in config.keys() else torch.nn.MSELoss()
        self.act_fn = config['act_fn'] if config['act_fn'] is not None else Identity()
        self.patience = config['patience'] if 'patience' in config.keys() else 5
        self.tol = config['tol'] if 'tol' in config.keys() else 1.0
        self.weight_samples = config['weight_samples'] if 'weight_samples' in config.keys() else False

        if config['norm_fn'] is None:
            self.norm_fn = Identity()
        else:
            self.norm_fn = config['norm_fn']
            self.norm_fn.reset_parameters() # needed for hp tuning

        if config['input_norm_fn'] is None:
            self.input_norm_fn = Identity()
        else:
            self.input_norm_fn = config['input_norm_fn']
            self.input_norm_fn.reset_parameters()

        
    def forward(self, x):
        x = self.input_norm_fn(x)
        y_hat = self.norm_fn(self.act_fn(self.strf(x)))
        
        return y_hat
    
    
    def configure_optimizers(self):
        return self.optim(self.parameters(), lr = self.lr)
    
    
    def get_penalty(self):
        L1, L2 = 0, 0
        for name, param in self.named_parameters():
            if 'norm' in name or name == 'strf.weight':
                L1 = L1 + torch.sum(torch.abs(param))
                L2 = L2 + torch.sum(param ** 2)
        
        L1 = L1 * self.alpha
        L2 = torch.sqrt(L2) * (1 - self.alpha)

        return self.lambd * (L1 + L2)


    def _prepare_batch(self, batch):
        x, y = batch
        y = y[-1]
        batch_size = y.shape[0]
        
        if len(x) > 1:
            x = torch.cat(x, 1)
        else:
            x = x[0]
            
        x = x.reshape([batch_size, -1])

        return x, y


    def _get_loss(self, batch):
        x, y = self._prepare_batch(batch)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        if self.weight_samples:
            mean_response = torch.mean(y, 0)
            loss_weights = 1 + (y - mean_response) ** 2
            loss = torch.mean(loss * loss_weights)
        
        return loss

    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch) + self.get_penalty()
        self.log('train_loss', loss)

        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_loss', loss)

        return loss


    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test_loss', loss)

        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        x, y = self._prepare_batch(batch)
        
        return self(x)