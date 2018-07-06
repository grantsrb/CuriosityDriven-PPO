import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

def cuda_if(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj

class Updater():
    """
    This class converts the data collected from the rollouts into useable data to update
    the model. The main function to use is calc_loss which accepts a rollout to
    add to the global loss of the model. The model isn't updated, however, until calling
    calc_gradients followed by update_model. If the size of the epoch is restricted by the memory, you can call calc_gradients to clear the graph.
    """

    def __init__(self, net, hyps): 
        self.net = net 
        self.old_net = copy.deepcopy(self.net) 
        self.hyps = hyps
        self.gamma = hyps['gamma']
        self.lambda_ = hyps['lambda_']
        self.use_nstep_rets = hyps['use_nstep_rets']
        self.optim_type = hyps['optim_type']
        self.optim = self.new_optim(hyps['lr'])    
        self.cache = None

        # Tracking variables
        self.info = {}
        self.max_adv = -1
        self.min_adv = 1
        self.max_rew = -1e15
        self.min_rew = 1e15
        self.max_minsurr = -1e10
        self.min_minsurr = 1e10

    def update_model(self, shared_data):
        """
        This function accepts the data collected from a rollout and performs PPO update iterations
        on the neural net.

        datas - dict of torch tensors with shared memory to collect data. Each 
                tensor contains indices from idx*n_tsteps to (idx+1)*n_tsteps
                Keys (assume string keys):
                    "states" - MDP states at each timestep t
                            type: FloatTensor
                            shape: (n_states, *state_shape)
                    "next_states" - MDP states at timestep t+1
                            type: FloatTensor
                            shape: (n_states, *state_shape)
                    "dones" - Collects the dones collected at each timestep t
                            type: FloatTensor
                            shape: (n_states,)
                    "actions" - Collects actions performed at each timestep t
                            type: LongTensor
                            shape: (n_states,)
        """
        hyps = self.hyps
        states = shared_data['states']
        next_states = shared_data['next_states']
        actions = shared_data['actions']
        dones = shared_data['dones']

        self.update_cache(shared_data)

        # Make rewards
        self.net.req_grads(False)
        embs = self.net.embeddings(Variable(states))
        one_hot_actions = cuda_if(self.one_hot_encode(actions, self.net.output_space))
        fwd_inputs = torch.cat([embs.data, one_hot_actions], dim=-1)
        fwd_preds = self.net.fwd_dynamics(Variable(fwd_inputs))
        del fwd_inputs
        targets = torch.cat([embs.data[1:], self.net.embeddings(Variable(next_states[-1:])).data], dim=0)
        del embs
        rewards = F.mse_loss(fwd_preds, targets, size_average=False, reduce=False)
        rewards = rewards.view(len(fwd_preds),-1).mean(-1).data
        del targets
        del fwd_preds
        if hyps['norm_rews']:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        self.max_rew = max(rewards.max().item(), self.max_rew)
        self.min_rew = min(rewards.min().item(), self.min_rew)

        if hyps['use_gae']:
            advantages, returns = self.make_advs_and_rets(states,next_states,rewards,dones)
        else:
            advantages = self.discount(rewards, dones, hyps['gamma'])
            returns = advantages

        if hyps['norm_advs']:
            advantages = (advantages - advantages.mean())/(advantages.std()+1e-6)

        avg_epoch_loss,avg_epoch_policy_loss,avg_epoch_val_loss,avg_epoch_entropy = 0,0,0,0
        avg_epoch_dyn_loss = 0
        self.net.train(mode=True)
        self.net.req_grads(True)
        self.old_net.load_state_dict(self.net.state_dict())
        self.old_net.train(mode=True)
        self.old_net.req_grads(False)
        self.optim.zero_grad()
        for epoch in range(hyps['n_epochs']):

            loss, epoch_loss, epoch_policy_loss, epoch_val_loss, epoch_entropy = 0,0,0,0,0
            epoch_dyn_loss = 0
            indices = torch.randperm(len(states)).long()
            if self.cache is not None:
                cache_len = len(self.cache['states'])
                cache_idxs = torch.randperm(cache_len).long()

            for i in range(len(indices)//hyps['batch_size']):

                # Get data for batch
                startdx = i*hyps['batch_size']
                endx = (i+1)*hyps['batch_size']
                idxs = indices[startdx:endx]
                batch_data = states[idxs],next_states[idxs],actions[idxs],advantages[idxs],returns[idxs]

                # Optional forward dynamics memory replay
                if self.cache is not None and i*hyps['cache_batch'] < cache_len:
                    cachxs = cache_idxs[i*hyps['cache_batch']:(i+1)*hyps['cache_batch']]
                    cache_states = self.cache['states'][cachxs]
                    cache_next_states = self.cache['next_states'][cachxs]
                    cache_actions = self.cache['actions'][cachxs]
                    cache_batch = cache_states, cache_next_states, cache_actions
                    cache_loss = hyps['cache_coef']*self.cache_losses(*cache_batch)
                else:
                    cache_loss = Variable(torch.zeros(1))

                # Total Loss
                policy_loss, val_loss, entropy, dyn_loss = self.ppo_losses(*batch_data)
                ppo_term = (1-self.hyps['dyn_coef'])*(policy_loss + val_loss - entropy)
                dynamics_term = self.hyps['dyn_coef']*(dyn_loss + cache_loss)
                loss = ppo_term + dynamics_term

                # Gradient Step
                loss.backward()
                self.norm = nn.utils.clip_grad_norm_(self.net.parameters(), hyps['max_norm'])
                self.optim.step()
                self.optim.zero_grad()
                epoch_loss += float(loss.data)
                epoch_policy_loss += float(policy_loss.data)
                epoch_val_loss += float(val_loss.data)
                epoch_dyn_loss += float(dyn_loss.data)
                epoch_entropy += float(entropy.data)

            avg_epoch_loss += epoch_loss/hyps['n_epochs']
            avg_epoch_policy_loss += epoch_policy_loss/hyps['n_epochs']
            avg_epoch_val_loss += epoch_val_loss/hyps['n_epochs']
            avg_epoch_entropy += epoch_entropy/hyps['n_epochs']
            avg_epoch_dyn_loss += epoch_dyn_loss/hyps['n_epochs']

        self.info = {"Loss":float(avg_epoch_loss), 
                    "PiLoss":float(avg_epoch_policy_loss), 
                    "VLoss":float(avg_epoch_val_loss), 
                    "Entr":float(avg_epoch_entropy), 
                    "DynLoss":float(avg_epoch_entropy), 
                    "MaxAdv":float(self.max_adv),
                    "MinAdv":float(self.min_adv), 
                    "MinSurr":float(self.min_minsurr), 
                    "MaxSurr":float(self.max_minsurr),
                    "MaxRew":float(self.max_rew),
                    "MinRew":float(self.min_rew)} 
        self.max_adv, self.min_adv, = -1, 1
        self.max_minsurr, self.min_minsurr = -1e10, 1e10
        self.max_rew, self.min_rew = -1e15, 1e15

    def cache_losses(self, states, next_states, actions):
        """
        Creates a loss term from historical data to avoid forgetting within the forward
        dynamics model.

        states - torch FloatTensor minibatch of states with shape (batch_size, C, H, W)
        next_states - torch FloatTensor minibatch of next states with shape (batch_size, C, H, W)
        actions - torch LongTensor minibatch of empirical actions with shape (batch_size,)
        """
        
        embs = self.net.embeddings(Variable(states))
        embs.detach()
        one_hot_actions = self.one_hot_encode(actions.long(), self.net.output_space)
        fwd_inputs = torch.cat([embs.data, cuda_if(one_hot_actions)], dim=-1)
        fwd_preds = self.net.fwd_dynamics(Variable(fwd_inputs))
        fwd_targs = self.net.embeddings(Variable(next_states))
        fwd_targs.detach()
        dynamics_loss = F.mse_loss(fwd_preds, Variable(fwd_targs.data))
        return dynamics_loss


    def ppo_losses(self, states, next_states, actions, advs, rets):
        """
        Completes the ppo specific loss approach

        states - torch FloatTensor minibatch of states with shape (batch_size, C, H, W)
        next_states - torch FloatTensor minibatch of next states with shape (batch_size, C, H, W)
        actions - torch LongTensor minibatch of empirical actions with shape (batch_size,)
        advs - torch FloatTensor minibatch of empirical advantages with shape (batch_size,)
        rets - torch FloatTensor minibatch of empirical returns with shape (batch_size,)

        Returns:
            policy_loss - the PPO CLIP policy gradient shape (1,)
            val_loss - the critic loss shape (1,)
            entropy - the entropy of the action predictions shape (1,)
            dynamics_loss 
        """
        hyps = self.hyps

        # Get new Outputs
        embs = self.net.embeddings(Variable(states))
        vals, raw_pis = self.net.val_pi(embs)
        probs = F.softmax(raw_pis, dim=-1)
        pis = probs[cuda_if(torch.arange(0,len(probs)).long()), actions]

        # Get old Outputs
        old_vals, old_raw_pis = self.old_net.forward(Variable(states))
        old_vals.detach(), old_raw_pis.detach()
        old_probs = F.softmax(old_raw_pis, dim=-1)
        old_pis = old_probs[cuda_if(torch.arange(0,len(old_probs))).long(), actions]

        # Policy Loss
        if hyps['norm_batch_advs']:
            advs = (advs - advs.mean())
            advs = advs / (advs.std() + 1e-7)
        self.max_adv = max(torch.max(advs), self.max_adv) # Tracking variable
        self.min_adv = min(torch.min(advs), self.min_adv) # Tracking variable
        advs = Variable(advs)
        ratio = pis/(old_pis+1e-5)
        surrogate1 = ratio*advs
        surrogate2 = torch.clamp(ratio, 1.-hyps['epsilon'], 1.+hyps['epsilon'])*advs
        min_surr = torch.min(surrogate1, surrogate2)
        self.max_minsurr = max(torch.max(min_surr.data), self.max_minsurr)
        self.min_minsurr = min(torch.min(min_surr.data), self.min_minsurr)
        policy_loss = -min_surr.mean()

        # Value loss
        if hyps['use_gae']:
            rets = Variable(rets)
            if hyps['clip_vals']:
                clipped_vals = old_vals + torch.clamp(vals-old_vals, -hyps['epsilon'], hyps['epsilon'])
                v1 = .5*(vals.squeeze()-rets)**2
                v2 = .5*(clipped_vals.squeeze()-rets)**2
                val_loss = hyps['val_const'] * torch.max(v1,v2).mean()
            else:
                val_loss = hyps['val_const'] * F.mse_loss(vals.squeeze(), rets)
        else:
            val_loss = Variable(cuda_if(torch.zeros(1)))

        # Entropy Loss
        softlogs = F.log_softmax(raw_pis, dim=-1)
        entropy_step = torch.sum(softlogs*probs, dim=-1)
        entropy = -hyps['entr_coef'] * torch.mean(entropy_step)

        # Dynamics Loss
        one_hot_actions = self.one_hot_encode(actions, self.net.output_space)
        fwd_inputs = torch.cat([embs.data, cuda_if(one_hot_actions)], dim=-1)
        fwd_preds = self.net.fwd_dynamics(Variable(fwd_inputs))
        fwd_targs = self.net.embeddings(Variable(next_states)).data
        dynamics_loss = F.mse_loss(fwd_preds, Variable(fwd_targs))

        return policy_loss, val_loss, entropy, dynamics_loss

    def one_hot_encode(self, idxs, width):
        """
        Creates one hot encoded vector of the inputs.

        idxs - torch LongTensor of indexes to be converted to one hot vectors.
            type: torch LongTensor
            shape: (n_entries,)
        width - integer of the size of each one hot vector
            type: int
        """
        
        one_hots = torch.zeros(len(idxs), width)
        one_hots[torch.arange(0,len(idxs)).long(), idxs] = 1
        return one_hots

    def make_advs_and_rets(self, states, next_states, rewards, dones):
        """
        Creates the advantages and returns.

        states - torch FloatTensor of shape (L, C, H, W)
        next_states - torch FloatTensor of shape (L, C, H, W)
        rewards - torch FloatTensor of empirical rewards (L,)

        Returns:
            advantages - torch FloatTensor of shape (L,)
            returns - torch FloatTensor of shape (L,)
        """

        self.net.req_grads(False)
        vals, raw_pis = self.net.forward(Variable(states))
        next_vals, _ = self.net.forward(Variable(next_states))
        self.net.req_grads(True)

        # Make Advantages
        advantages = self.gae(rewards.squeeze(), vals.data.squeeze(), next_vals.data.squeeze(), dones.squeeze(), self.gamma, self.lambda_)

        # Make Returns
        if self.use_nstep_rets: 
            returns = advantages + vals.data.squeeze()
        else: 
            rewards[dones==1] += self.hyps['gamma']*next_vals[dones==1] # Include bootstrap
            returns = self.discount(rewards.squeeze(), dones.squeeze(), self.gamma)

        return advantages, returns

    def gae(self, rewards, values, next_vals, dones, gamma, lambda_):
        """
        Performs Generalized Advantage Estimation

        rewards - torch FloatTensor of actual rewards collected. Size = L
        values - torch FloatTensor of value predictions. Size = L
        next_vals - torch FloatTensor of value predictions. Size = L
        dones - torch FloatTensor of done signals. Size = L
        gamma - float discount factor
        lambda_ - float gae moving average factor

        Returns
         advantages - torch FloatTensor of genralized advantage estimations. Size = L
        """
        deltas = rewards + gamma*next_vals - values
        del next_vals
        return self.discount(deltas, dones, gamma*lambda_)

    def discount(self, array, dones, discount_factor):
        """
        Dicounts the argued array following the bellman equation.

        array - array to be discounted
        dones - binary array denoting the end of an episode
        discount_factor - float between 0 and 1 used to discount the reward

        Returns the discounted array as a torch FloatTensor
        """
        running_sum = 0
        discounts = cuda_if(torch.zeros(len(array)))
        for i in reversed(range(len(array))):
            if dones[i] == 1: running_sum = 0
            running_sum = array[i] + discount_factor*running_sum
            discounts[i] = running_sum
        return discounts

    def update_cache(self, shared_data):
        keys = ['actions', 'states', 'next_states']
        if self.cache is None:
            self.cache = dict()
            for key in keys:
                self.cache[key] = shared_data[key].clone()
        else:
            cache_size = self.hyps['cache_size']
            if len(self.cache[keys[0]]) < cache_size:
                for key in keys:
                    cache = self.cache[key]
                    new_data = shared_data[key]
                    self.cache[key] = torch.cat([cache, new_data], dim=0)
            else:
                n_cache_refresh = self.hyps['n_cache_refresh']
                for key in keys:
                    cache = self.cache[key]
                    new_data = shared_data[key]
                    cache_perm = torch.randperm(len(cache)).long()
                    cache_idxs = cache_perm[:n_cache_refresh]
                    data_perm = torch.randperm(len(new_data)).long()
                    data_idxs = data_perm[:n_cache_refresh]
                    self.cache[key][cache_idxs] = new_data[data_idxs]

    def print_statistics(self):
        print(" – ".join([key+": "+str(round(val,5)) for key,val in sorted(self.info.items())]))

    def log_statistics(self, log, T, reward, avg_action, best_avg_rew):
        log.write("Step:"+str(T)+" – "+" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]+["EpRew: "+str(reward), "AvgAction: "+str(avg_action), "BestRew:"+str(best_avg_rew)]) + '\n')
        log.flush()

    def save_model(self, net_file_name, optim_file_name):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        if optim_file_name is not None:
            torch.save(self.optim.state_dict(), optim_file_name)
    
    def new_lr(self, new_lr):
        new_optim = self.new_optim(new_lr)
        new_optim.load_state_dict(self.optim.state_dict())
        self.optim = new_optim

    def new_optim(self, lr):
        if self.optim_type == 'rmsprop':
            new_optim = optim.RMSprop(self.net.parameters(), lr=lr) 
        elif self.optim_type == 'adam':
            new_optim = optim.Adam(self.net.parameters(), lr=lr) 
        else:
            new_optim = optim.RMSprop(self.net.parameters(), lr=lr) 
        return new_optim
