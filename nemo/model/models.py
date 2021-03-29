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
        
        self.strf = torch.nn.Linear(
            self.in_h * self.in_w * self.n_frames, 
            self.n_neurons
        )

        self.callbacks = [
            EarlyStopping(
                monitor = 'val_loss',
                patience = 4,
                min_delta = 1e-4,
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
        self.act_fn = config['act_fn'] if 'act_fn' in config.keys() else Identity()
        self.norm_fn = config['norm_fn'] if 'norm_fn' in config.keys() else Identity()

        
    def forward(self, x):
        x = (x - torch.mean(x, 0)) / (torch.std(x, 0) + 1e-8)
        y_hat = self.norm_fn(self.strf(x))
        
        return self.act_fn(y_hat)
    
    
    def configure_optimizers(self):
        return self.optim(self.parameters(), lr = self.lr)
    
    
    def get_penalty(self):
        l1 = self.strf.weight.norm(1) * self.alpha
        l2 = self.strf.weight.norm(2) * (1 - self.alpha)

        return self.lambd * (l1 + l2)


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

        return self.loss_fn(y_hat, y) 

    
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