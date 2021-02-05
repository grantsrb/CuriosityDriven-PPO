import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rnnlocator import MediumCNN, SimpleCNN, Flatten
from transformer.models import Attncoder


class ConvEmbedder(nn.Module):
    def __init__(self, input_space, h_size=200, bnorm=False,
                                                lnorm=False,
                                                **kwargs):
        super().__init__()
        self.input_space = input_space
        self.h_size = h_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        
        self.features = MediumCNN(img_shape=self.input_space,
                                  emb_size=self.h_size,
                                  feat_bnorm=self.bnorm)
        self.flat_size = int(self.features.seq_len*self.h_size)
        print("Flat Features Size:", self.flat_size)

        block = [Flatten()]
        if self.lnorm:
            bloc.append(nn.LayerNorm(self.flat_size))
        block.append(nn.Linear(self.flat_size, self.h_size))
        self.resize_emb = nn.Sequential(*block)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x: float tensor (B, C, H, W)

        Outputs:
            FloatTensor (B,E)
        """
        feats = self.features(x)
        return self.resize_emb(feats)

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

class Embedder(nn.Module):
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, h_size=200, bnorm=False, 
                                                lnorm=False,
                                                **kwargs):
        super(Embedder, self).__init__()

        self.input_space = input_space
        self.h_size = h_size
        self.bnorm = bnorm
        self.lnorm = lnorm

        self.convs = nn.ModuleList([])

        # Embedding Net
        shape = input_space.copy()

        ksize=3; stride=1; padding=1; out_depth=16
        self.convs.append(self.conv_block(input_space[-3],out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=32
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=48
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=64
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)
        
        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        print("Flat Features Size:", self.flat_size)

        block = []
        if self.lnorm:
            block = [nn.LayerNorm(self.flat_size)]
        block.append(nn.Linear(self.flat_size, self.h_size))
        #block.append(nn.ReLU())
        self.resize_emb = nn.Sequential(*block)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding, stride))
        return new_shape
        
    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def forward(self, state, *args, **kwargs):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor with shape (BatchSize, Channels, Height, Width)
        """
        feats = self.features(state)
        feats = feats.view(feats.shape[0], -1)
        state_embs = self.resize_emb(feats)
        return state_embs

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="lerelu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride, padding=padding))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU(negative_slope=.05))
        elif "selu" in activation:
            block.append(nn.SELU())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if bnorm:
            block.append(nn.BatchNorm1d(chan_out))
        return nn.Sequential(*block)

    def add_noise(self, x, mean=0.0, std=0.01):
        """
        Adds a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0.0:
            means = means + mean
        noise = self.cuda_if(torch.normal(means,std=std))
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x+noise

    def multiply_noise(self, x, mean=1, std=0.01):
        """
        Multiplies a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0:
            means = means + mean
        noise = self.cuda_if(torch.normal(means,std=std))
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x*noise

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

class FCEmbedder(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, h_size=200, bnorm=False,
                                                lnorm=False,
                                                **kwargs):
        super().__init__()

        self.input_space = input_space
        self.flat_size = int(np.prod(input_space))
        self.h_size = h_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        
        modules = []
        modules.append(nn.Linear(self.flat_size,h_size))
        if self.bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(h_size, h_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.ReLU())
        if self.lnorm:
            modules.append(nn.LayerNorm(self.h_size))
        modules.append(nn.Linear(self.h_size,h_size))
        #modules.append(nn.ReLU())

        self.features = nn.Sequential(*modules)

    def forward(self, state, *args, **kwargs):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor (BatchSize, Channels, Height, Width)
        """
        return self.features(state.reshape(len(state),-1))

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

class Ensemble(nn.Module):
    def __init__(self, net, n_nets=3):
        """
        net: Module
            the architecture to use for the ensemble
        n_nets: int
            the total number of networks in the ensemble
        """
        super().__init__()
        self.nets = nn.ModuleList([])
        for _ in range(n_copies):
            new_net = copy.deepcopy(net)
            for name,modu in new_net.named_modules():
                if isinstance(modu, nn.Conv2d) or isinstance(modu, nn.Linear):
                    nn.init.xavier_uniform(modu.weight)
            self.nets.append(new_net)

    def forward(self, x, h=None, **kwargs):
        preds = []
        for net in self.nets:
            preds.append(net(x,h))
        return preds

class CatModule(nn.Module):
    def __init__(self, modu):
        super().__init__()
        self.modu = modu

    def forward(self, x, h):
        """
        x: FloatTensor (B,X)
        h: FloatTensor (B,H)
        """
        inpt = torch.cat([x,h],dim=-1)
        return self.modu(inpt)

class RecurrentFwdModel(nn.Module):
    def __init__(self, h_size, a_size, rnn_type="GRUCell",
                                       lnorm=False):
        """
        h_size: int
        a_size: int
            size of action space
        rnn_type: str
            the name of the rnn cell type from pytorch
        lnorm: bool
            if true, the hidden vector is layer normed before making
            prediction
        """
        super().__init__()
        self.h_size = h_size
        self.a_size = a_size
        self.lnorm = lnorm
        self.rnn = getattr(nn, rnn_type)(input_size=self.h_size+self.a_size,
                                         hidden_size=self.h_size)
        self.dynamics = []
        if self.lnorm: self.dynamics.append(nn.LayerNorm(self.h_size))
        self.dynamics += [nn.Linear(h_size, h_size), 
            nn.ReLU(), nn.Linear(h_size, h_size)]
        self.dynamics = nn.Sequential(*self.dynamics)

        # Learned initialization for rnn hidden vector
        self.h_shape = (1,self.h_size)
        self.h_init = torch.randn(self.h_shape)
        divisor = float(np.sqrt(self.h_size))
        self.h_init = nn.Parameter(self.h_init/divisor)

    def fresh_h(self, batch_size=1):
        """
        returns an h that is of shape (B,E)
        """
        return self.h_init.repeat(batch_size,1)

    def forward(self, x, h):
        """
        x: float tensor (B, E+A)
        h: float tensor (B, E)
        """
        h = self.rnn(x,h)
        pred = self.dynamics(h)
        return pred,h

class FwdRunModel(nn.Module):
    def __init__(self, embedder, dynamics):
        """
        embedder: module
            takes an x or an x and h vector and outputs features
        dynamics: module
            takes an input of features concatenated with actions and an
            h vector and outputs feature prediction and a new h
        """
        super().__init__()
        self.embedder = embedder
        self.dynamics = dynamics
        self.emb_size = embedder.h_size

    def fresh_h(self, batch_size=1):
        self.h = self.dynamics.fresh_h(batch_size)
        return self.h

    def forward(self, x, a, h=None, ret_embs=False):
        """
        x: float tensor (B, C, H, W)
        a: float tensor (B, A)
        h: float tensor (B, E)
        """
        if h is None:
            h = self.h
        embs = self.embedder(x, h) # (B, E)
        feats = torch.cat([embs,a], dim=-1) # (B, E+A)
        pred, h = self.dynamics(feats, h) # (B, E), (B, E)
        if ret_embs:
            return pred, h, embs
        return pred, h

class ContrastModel(nn.Module):
    def __init__(self, emb_size=512, out_size=2, *args, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.out_size = out_size
        self.attn = Attncoder(1, *args, emb_size=self.emb_size,
                                 **kwargs,
                                 use_mask=False,
                                 init_decs=False,
                                 gen_decs=False)
        self.classifier = nn.Sequential(
                    nn.Linear(self.emb_size, self.emb_size),
                    nn.ReLU(),
                    nn.Linear(self.emb_size, self.out_size))

    def forward(self, x, y):
        feats = self.attn(x,y) # (B,S,E)
        shape = feats.shape
        outs = self.classifier(feats.reshape(-1,self.emb_size)) #(B*S,O)
        return outs.reshape(*shape[:-1], self.out_size) # (B,S,O)
