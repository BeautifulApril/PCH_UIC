import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import ipdb
import math

class WSLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(WSLinear, self).__init__(in_features, out_features, bias=bias)
    
    def forward(self, x):
        C_in = self.weight.size(1)
        temp_mean = self.weight.mean(dim = 1).view(-1,1)
        temp_std = self.weight.std(dim = 1).view(-1,1) + 1e-6
        re_weight = (self.weight - temp_mean) / temp_std / math.sqrt(C_in)
        return F.linear(x, re_weight, self.bias)
        

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims,  output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            # WSLinear(input_dims, output_dims),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dims, hidden_dims),
            # nn.ReLU(),
            # nn.Linear(hidden_dims, output_dims),
            # nn.LogSoftmax()
            # nn.Sigmoid()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class Generator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims,  output_dims):
        """Init discriminator."""
        super(Generator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dims, hidden_dims),
            # nn.ReLU(),
            # nn.Linear(hidden_dims, output_dims),
            # nn.LogSoftmax()
            # nn.Sigmoid()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(reduce=False, size_average=False)
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
            # self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # ipdb.set_trace()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
def set_requires_grad( nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
