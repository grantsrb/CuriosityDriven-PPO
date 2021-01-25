from utils import next_state, sample_action, cuda_if
from torch.autograd import Variable
import torch
import gym
import torch.nn.functional as F
from collections import deque
import numpy as np
from ml_utils.utils import try_key
from environments import SeqEnv

class Runner:
    def __init__(self, datas, hyps, gate_q, stop_q, end_q, rew_q):
        """
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
                    "preprocess" - function to preprocess raw observations
                    "env_type" - type of gym environment to be interacted 
                                with. Follows OpenAI's gym api.
        datas - dict of torch tensors with shared memory to collect data. Each 
                tensor contains indices from idx*n_tsteps to (idx+1)*n_tsteps
                Keys (assume string keys):
                    "states" - Collects the MDP states at each timestep t
                    "next_states" - Collects the MDP states at timestep t+1
                    "dones" - Collects the dones collected at each timestep t
                    "actions" - Collects actions performed at each timestep t
        gate_q - multiprocessing queue. Allows main process to control when
                rollouts should be collected.
        stop_q - multiprocessing queue. Used to indicate to main process that
                a rollout has been collected.
        end_q - multiprocessing queue. Used to indicate that
                the experiment is finished
        rew_q - holds average reward over all processes
                type: multiprocessing Queue
        """

        self.hyps = hyps
        self.datas = datas
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.end_q = end_q
        self.rew_q = rew_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])

    def run(self, net):
        """
        run is the entry function to begin collecting rollouts from the
        environment using the specified net. gate_q indicates when to begin
        collecting a rollout and is controlled from the main process.
        The stop_q is used to indicate to the main process that a new rollout
        has been collected.

        net - torch Module object. This is the model to interact with the
            environment.
        """
        self.net = net
        float_params = try_key(self.hyps,"float_params", dict())
        self.env = SeqEnv(self.hyps['env_type'], self.hyps['seed'],
                                            worker_id=None,
                                            float_params=float_params)
        state = next_state(self.env, self.obs_deque, obs=None, reset=True, 
                                            preprocess=self.hyps['preprocess']) 
        self.state_bookmark = state
        self.ep_rew = 0
        self.net.train(mode=False) # fixes batchnorm issues
        for p in self.net.parameters(): # Turn off gradient collection
            p.requires_grad = False
        with torch.no_grad():
            while self.end_q.empty():
                idx = self.gate_q.get() # Opened from main process
                self.rollout(self.net, idx, self.hyps)
                self.stop_q.put(idx) # Signals to main process that data has been collected

    def rollout(self, net, idx, hyps):
        """
        rollout handles the actual rollout of the environment for n steps in time.
        It is called from run and performs a single rollout, placing the
        collected data into the shared lists found in the datas dict.

        net - torch Module object. This is the model to interact with the
            environment.
        idx - int identification number distinguishing the
            portion of the shared array designated for this runner
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
                    "preprocess" - function to preprocess raw observations
        """
        hyps = self.hyps
        state = self.state_bookmark
        n_tsteps = hyps['n_tsteps']
        startx = idx*n_tsteps
        for i in range(n_tsteps):
            self.datas['states'][startx+i] = cuda_if(torch.FloatTensor(state))
            val, logits = net(self.datas['states'][startx+i][None])
            if self.hyps['discrete_env']:
                probs = F.softmax(logits, dim=-1)
                action = sample_action(probs.data)
                action = int(action.item())
            else:
                mu,sig = logits
                action = mu + torch.randn_like(sig)*sig
                action = action.cpu().detach().numpy().squeeze()
                if len(action.shape) == 0:
                    action = np.asarray([float(action)])
            obs, rew, done, info = self.env.step(action+hyps['action_shift'])
            if hyps['render']:
                self.env.render()
            self.ep_rew += rew
            reset = done
            if "Pong" in hyps['env_type'] and rew != 0:
                done = True
            if done:
                self.rew_q.put(.99*self.rew_q.get() + .01*self.ep_rew)
                self.ep_rew = 0

            self.datas['dones'][startx+i] = 0
            if isinstance(action, np.ndarray):
                action = cuda_if(torch.from_numpy(action))
            self.datas['actions'][startx+i] = action
            state = next_state(self.env, self.obs_deque,
                                         obs=obs,
                                         reset=reset, 
                                         preprocess=hyps['preprocess'])
            if i > 0:
                self.datas['next_states'][startx+i-1] = self.datas['states'][startx+i]

        endx = startx+n_tsteps-1
        self.datas['next_states'][endx] = cuda_if(torch.FloatTensor(state))
        self.datas['dones'][endx] = 1.
        self.state_bookmark = state
