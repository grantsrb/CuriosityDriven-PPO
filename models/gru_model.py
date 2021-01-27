import torch
from torch.autograd import Variable
from .embedder import Embedder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Simple, sequential convolutional net.
'''

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

class GRUModel(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, output_space, h_size=200,
                                                  bnorm=False,
                                                  lnorm=False,
                                                  discrete_env=True):
        super().__init__()

        self.input_space = input_space
        self.output_space = output_space
        self.h_size = h_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.discrete_env = discrete_env

        # Embedding Net
        self.embedder = Embedder(self.input_space, self.h_size,
                                                   self.bnorm)

        # Recurrent Network
        self.h_init = torch.randn(1,self.h_size)
        divisor = float(np.sqrt(self.h_size))
        self.h_init = nn.Parameter(self.h_init/divisor)
        self.rnn = nn.GRUCell(input_size=(self.h_size+self.output_space),
                              hidden_size=self.h_size)

        # Policy
        self.pre_valpi = CatModule(nn.Sequential(
                            nn.Linear(2*self.h_size, self.h_size),
                            nn.ReLU()))
        self.pi = nn.Linear(self.h_size, self.output_space)
        if not self.discrete_env:
            self.logsigs = nn.Parameter(torch.zeros(1,self.output_space))
        self.value = nn.Linear(self.h_size, 1)

        if self.lnorm:
            self.layer_norm = nn.LayerNorm(self.h_size)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding, stride))
        return new_shape
        
    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def embeddings(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor (BatchSize, Channels, Height, Width)
        """
        return self.embedder(state)

    def forward(self, x, h):
        embs = self.embeddings(x)
        val, pi, h = self.val_pi(embs, h)
        return val, pi, h

    def val_pi(self, state_emb, h):
        """
        Uses the state embedding to produce an action.

        state_emb - the state embedding created by the emb_net
        h - the recurrent hidden state
        """
        state_emb = self.pre_valpi(state_emb, h)
        pi = self.pi(state_emb)
        value = self.value(state_emb)
        rnn_inpt = torch.cat([state_emb, pi], dim=-1)
        h = self.rnn(rnn_inpt, h)
        if self.lnorm:
            h = self.layer_norm(h)
        if not self.discrete_env:
            sig = torch.exp(self.logsigs)+0.00001
            sig = sig.repeat(len(pi),1)
            mu = torch.tanh(pi)
            return value, (mu,sig), h
        return value, pi, h

    def fresh_h(self, batch_size=1):
        """
        returns a new hidden state vector for the rnn
        """
        return self.h_init.repeat(batch_size,1)

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

