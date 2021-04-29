import torch


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class SpatioTemporalConv(torch.nn.Module):
    ''' PyTorch Layer that does 3D convolution with fixed weight tensor. ''' 

    def __init__(self, fbank, stride):
        '''
        Args:
            fbank (np.ndarray): Weight tensor of shape out_c, in_c, kd, kh, kw.
            stride (int): The stride of the kernel in the height and width dims.
        '''

        super(SpatioTemporalGaborConv, self).__init__()
        
        if len(fbank.shape) != 5:
            raise ValueError('fbank should have 5 dims (out_c, in_c, kd, kh, kw).')

        self.fbank = torch.FloatTensor(fbank)
        self.stride = stride

    def forward(self, x):
        return torch.nn.functional.conv3d(
            x,
            self.fbank,
            padding = (0, (self.fbank.shape[-2] - 1) // 2, (self.fbank.shape[-1] - 1) // 2),
            stride = self.stride
        )