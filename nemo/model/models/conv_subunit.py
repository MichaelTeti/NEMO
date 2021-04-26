import numpy as np
import torch


class ElasticNetGaborSubunit(LightningModule):
    def __init__(self, config):
        ''' 3D Convolutional Elastic Net built in PyTorch '''
        
        super().__init__()
        self._set_hparams(config)

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError('alpha should be >= 0.0 and <= 1.0')

        if self.weight_samples:
            if self.loss_fn.reduction != 'none':
                raise ValueError('loss_fn should have reduction=none if weight_samples is True.')

        import numpy as np
        self.conv = GaborConv(
            [0.1],
            [0, 180, 234.9, 274.9, 45, 90],
            [0, np.pi / 2],
            [2, 3, 5, 7],
            [1.0],
            63
        )

        self.pool = torch.nn.Conv3d(
            self.conv.fbank.shape[0] * 2,
            self.n_neurons,
            kernel_size = (self.n_frames, self.in_h, self.in_w),
            padding = 0
        )

        self.scale = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad = True)
        self.slope = torch.nn.Parameter(torch.FloatTensor([-1.0]), requires_grad = True)
        self.horiz_shift = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad = True)
        self.vert_shift = 0 #torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad = True)

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
        self.conv_act_fn = config['conv_act_fn'] if config['conv_act_fn'] is not None else Identity()
        self.patience = config['patience'] if 'patience' in config.keys() else 5
        self.tol = config['tol'] if 'tol' in config.keys() else 1.0
        self.weight_samples = config['weight_samples'] if 'weight_samples' in config.keys() else False
        self.n_filters = config['n_filters'] if 'n_filters' in config.keys() else 32
        self.stride = config['stride'] if 'stride' in config.keys() else 1 
        self.kernel_size = config['kernel_size'] if 'kernel_size' in config.keys() else (9, 9, 9)
        self.out_act_fn = Logistic()

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
        x = self.norm_fn(self.conv(x))
        quad = x ** 2
        lin = torch.nn.functional.relu(x)
        x = torch.cat([lin, quad], 1)
        x = self.pool(x)
        y_hat = self.out_act_fn(
            torch.squeeze(x),
            self.scale,
            self.slope,
            self.horiz_shift,
            self.vert_shift
        )
        
        return y_hat
    
    
    def configure_optimizers(self):
        return self.optim(self.parameters(), lr = self.lr)
    
    
    def get_penalty(self):
        L1 = self.pool.weight.norm(1)
        L2 = self.pool.weight.norm(2)
        L1 = L1 * self.alpha
        L2 = L2 * (1 - self.alpha)

        return self.lambd * (L1 + L2)


    def _prepare_batch(self, batch):
        x, y = batch
        y = torch.squeeze(y[-1])
        x = torch.stack(x, 2)

        return x, y


    def _get_loss(self, batch):
        x, y = self._prepare_batch(batch)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        if self.weight_samples:
            mean_response = torch.mean(y, 0)
            loss_weights = 1 + torch.sqrt((y - mean_response) ** 2)
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