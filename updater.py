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

    def __init__(self, net, fwd_net, hyps, inv_net=None, recon_net=None): 
        self.net = net
        self.fwd_net = fwd_net
        self.old_net = copy.deepcopy(self.net).cpu()
        self.fwd_embedder = self.net.embedder
        if hyps['seperate_embs']:
            if "fwd_emb_model" in hyps and hyps['fwd_emb_model'] is not None:
                args = {**hyps}
                args['bnorm'] = hyps['fwd_bnorm']
                args['lnorm'] = hyps['fwd_lnorm']
                args['input_space'] = args['state_shape']
                self.fwd_embedder = cuda_if(hyps['fwd_emb_model'](**args))
            else:
                self.fwd_embedder = copy.deepcopy(self.net.embedder)
            if hyps['resume']:
                self.fwd_embedder.load_state_dict(torch.load(hyps['fwd_emb_file']))
        self.hyps = hyps
        self.inv_net = inv_net
        self.recon_net = recon_net
        self.gamma = hyps['gamma']
        self.lambda_ = hyps['lambda_']
        self.use_nstep_rets = hyps['use_nstep_rets']
        optim_type = hyps['optim_type']
        self.optim = self.new_optim(net.parameters(), hyps['lr'], optim_type)
        optim_type = hyps['fwd_optim_type']
        self.fwd_optim = self.new_optim(self.fwd_net.parameters(),
                                        hyps['fwd_lr'], optim_type)

        optim_type = hyps['reconinv_optim_type']
        params = list(self.fwd_embedder.parameters())
        make_optim = False
        if inv_net is not None:
            params = params + list(self.inv_net.parameters())
            make_optim = True
        if self.recon_net is not None:
            params = params + list(self.recon_net.parameters())
            make_optim = True
        if make_optim:
            self.reconinv_optim=self.new_optim(params, hyps['reconinv_lr'],
                                                       optim_type)
        self.cache = None

        # Tracking variables
        self.info = {}
        self.max_adv = -1
        self.min_adv = 1
        self.max_rew = -1e15
        self.min_rew = 1e15
        self.max_minsurr = -1e10
        self.min_minsurr = 1e10
        self.update_count = 0
        self.rew_mu  = None
        self.rew_sig = None

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
                    "hs" - Collects recurrent state vectors at each timestep t
                            type: FloatTensor
                            shape: (n_states,h_size)
        """
        torch.cuda.empty_cache()
        self.update_count += 1
        hyps = self.hyps
        states = shared_data['states']
        next_states = shared_data['next_states']
        actions = shared_data['actions']
        dones = shared_data['dones']
        hs = shared_data['hs']
        next_hs = shared_data['next_hs']

        ## Fwd Net BPTT
        #bptt_loss, fwd_hs = self.fwd_bptt(shared_data)

        cache_keys = ['actions', 'states', 'next_states', "hs", "next_hs"]
        self.update_cache(shared_data, cache_keys)


        # Make rewards
        self.net.req_grads(False)
        self.fwd_embedder.req_grads(False)
        self.fwd_embedder.eval()
        self.net.eval()
        with torch.no_grad():
            embs = self.fwd_embedder(states,hs)
            if self.hyps['discrete_env']:
                fwd_actions = cuda_if(self.one_hot_encode(actions,
                                           self.net.output_space))
            else:
                fwd_actions = actions
            fwd_inputs = torch.cat([embs.data, fwd_actions], dim=-1)
            if hyps['is_recurrent']:
                fwd_preds = self.fwd_net(fwd_inputs,hs.data)
            else:
                fwd_preds = self.fwd_net(Variable(fwd_inputs))
            del fwd_inputs
            #just adding the last targ emb to the already calculated embs
            targets = self.fwd_embedder(next_states, next_hs)
            del embs
            rewards = F.mse_loss(fwd_preds, targets, reduction="none")
            rewards = rewards.view(len(fwd_preds),-1).mean(-1).data
        self.fwd_embedder.train()
        self.net.train()
        # Bootstrapped value predictions are added within the
        # make_advs_and_rets fxn in the case of using the discounted
        # rewards for the returns
        del targets
        del fwd_preds
        if hyps['norm_rews'] or hyps['running_rew_norm']:
            if self.rew_mu is None or not hyps['running_rew_norm']:
                self.rew_mu,self.rew_sig = rewards.mean(),rewards.std()
            else:
                new_p = 1/self.update_count
                old_p = 1-new_p
                self.rew_mu = new_p*rewards.mean() + old_p*self.rew_mu
                self.rew_sig = new_p*rewards.std() + old_p*self.rew_sig
            rewards = (rewards - self.rew_mu) / (self.rew_sig + 1e-5)
        self.max_rew = max(rewards.max().item(), self.max_rew)
        self.min_rew = min(rewards.min().item(), self.min_rew)

        if hyps['use_gae']:
            with torch.no_grad():
                advantages, returns = self.make_advs_and_rets(states,
                                                          next_states,
                                                          rewards,
                                                          dones,
                                                          hs,
                                                          next_hs)
        else:
            # No bootstrapped values added when using this option
            advantages = self.discount(rewards, dones, hyps['gamma'])
            returns = advantages

        if hyps['norm_advs']:
            advantages = (advantages-advantages.mean())/\
                                  (advantages.std()+1e-6)

        avg_epoch_loss,avg_epoch_policy_loss,avg_epoch_val_loss,avg_epoch_entropy = 0,0,0,0
        avg_epoch_fwd_loss = 0
        avg_epoch_inv_loss = 0
        self.net.train(mode=True)
        self.net.req_grads(True)
        req_grad = self.inv_net is not None or self.recon_net is not None
        self.fwd_embedder.req_grads(req_grad)
        cuda_if(self.old_net).load_state_dict(self.net.state_dict())
        self.old_net.train()
        self.old_net.req_grads(False)
        with torch.no_grad():
            if self.hyps['is_recurrent']:
                old_vals,old_raw_pis,_ = self.old_net(states, h=hs)
            else:
                old_vals, old_raw_pis = self.old_net(states)
            self.old_net.cpu()
        sep_cache_loop = self.cache is not None and hyps['full_cache_loop']
        self.optim.zero_grad()
        for epoch in range(hyps['n_epochs']):
            epoch_cache_fwd = 0
            epoch_cache_inv = 0

            # Optional forward dynamics memory replay
            cache_len = len(self.cache['states'])
            cache_idxs = torch.randperm(cache_len).long()
            if sep_cache_loop:
                bsize = hyps['cache_batch']
                n_loops = cache_len//bsize
                for i in range(n_loops):
                    cachxs = cache_idxs[i*bsize:(i+1)*bsize]
                    cache_batch = []
                    cache_batch.append(self.cache['states'][cachxs])
                    cache_batch.append(self.cache['next_states'][cachxs])
                    cache_batch.append(self.cache['actions'][cachxs])
                    cache_batch.append(self.cache['hs'][cachxs])
                    cache_batch.append(self.cache['next_hs'][cachxs])
                    loss_tup = self.cache_losses(*cache_batch)
                    fwd_cache_loss,inv_cache_loss = loss_tup
                    loss = fwd_cache_loss + inv_cache_loss
                    loss.backward()
                    self.fwd_optim.step()
                    self.fwd_optim.zero_grad()
                    if self.inv_net is not None or self.recon_net is not None:
                        self.reconinv_optim.step()
                        self.reconinv_optim.zero_grad()
                    epoch_cache_fwd += fwd_cache_loss.item()
                    epoch_cache_inv += inv_cache_loss.item()
                epoch_cache_fwd /= n_loops
                epoch_cache_inv /= n_loops

            loss,epoch_loss,epoch_policy_loss,epoch_val_loss,epoch_entropy= 0,0,0,0,0
            epoch_fwd_loss = 0
            epoch_inv_loss = 0
            # PPO Loss
            indices = torch.randperm(len(states)).long()
            bsize = hyps['batch_size']
            cbsize = hyps['cache_batch']
            n_loops = len(indices)//hyps['batch_size']
            for i in range(n_loops):
                # Get data for batch
                startdx =  i*bsize
                endx = (i+1)*bsize
                idxs = indices[startdx:endx]

                if not sep_cache_loop and cache_len > (i+1)*cbsize:
                    cachxs = cache_idxs[i*cbsize:(i+1)*cbsize]
                    cache_batch = []
                    cache_batch.append(self.cache['states'][cachxs])
                    cache_batch.append(self.cache['next_states'][cachxs])
                    cache_batch.append(self.cache['actions'][cachxs])
                    cache_batch.append(self.cache['hs'][cachxs])
                    cache_batch.append(self.cache['next_hs'][cachxs])
                    loss_tup = self.cache_losses(*cache_batch)
                    fwd_cache_loss,inv_cache_loss = loss_tup
                else:
                    fwd_cache_loss = cuda_if(torch.zeros(1))
                    inv_cache_loss = cuda_if(torch.zeros(1))

                if not self.hyps['discrete_env']:
                    mu, sig = old_raw_pis
                    old_pis = (mu[idxs], sig[idxs])
                else:
                    old_pis = old_raw_pis[idxs]
                batch_data = [states[idxs],next_states[idxs],
                              actions[idxs],advantages[idxs],
                              returns[idxs],hs[idxs],next_hs[idxs],
                              old_vals[idxs],old_pis]
                # Total Loss
                loss_tup = self.ppo_losses(*batch_data)
                policy_loss,val_loss,entropy,fwd_loss,inv_loss = loss_tup

                ppo_term = (1-hyps['fwd_coef'])*(policy_loss + val_loss - entropy)
                fwd_term = hyps['fwd_coef']*fwd_loss + hyps['cache_coef']*fwd_cache_loss
                inv_term = hyps['inv_coef']*inv_loss + hyps['cache_coef']*inv_cache_loss
                loss = ppo_term + fwd_term + inv_loss
                #loss = ppo_term + fwd_term

                # Gradient Step
                loss.backward()
                if self.inv_net is not None:
                    _ = nn.utils.clip_grad_norm_(self.inv_net.parameters(),
                                                 hyps['max_norm'])
                if self.recon_net is not None:
                    _ = nn.utils.clip_grad_norm_(self.recon_net.parameters(),
                                                 hyps['max_norm'])
                self.norm = nn.utils.clip_grad_norm_(self.net.parameters(),
                                                 hyps['max_norm'])

                # Important to do fwd optim step first! This is because the fwd parameters are
                # in the regular optimizer as well
                self.fwd_optim.step()
                self.fwd_optim.zero_grad()
                self.optim.step()
                self.optim.zero_grad()
                if self.inv_net is not None or self.recon_net is not None:
                    self.reconinv_optim.step()
                    self.reconinv_optim.zero_grad()
                epoch_loss += float(loss.item())
                epoch_policy_loss += policy_loss.item()
                epoch_val_loss += val_loss.item()
                epoch_fwd_loss += fwd_term.item()
                epoch_inv_loss += inv_term.item()
                epoch_entropy += entropy.item()

            avg_epoch_loss +=        epoch_loss/hyps['n_epochs']/n_loops
            avg_epoch_policy_loss+=epoch_policy_loss/hyps['n_epochs']/n_loops
            avg_epoch_val_loss += epoch_val_loss/hyps['n_epochs']/n_loops
            avg_epoch_entropy +=  epoch_entropy/hyps['n_epochs']/n_loops
            avg_epoch_fwd_loss += epoch_fwd_loss/hyps['n_epochs']/2/n_loops
            avg_epoch_fwd_loss += epoch_cache_fwd/hyps['n_epochs']/2/n_loops
            avg_epoch_inv_loss += epoch_inv_loss/hyps['n_epochs']/2/n_loops
            avg_epoch_inv_loss += epoch_cache_inv/hyps['n_epochs']/2/n_loops

        self.info = {"Loss":float(avg_epoch_loss), 
                    "PiLoss":float(avg_epoch_policy_loss), 
                    "VLoss":float(avg_epoch_val_loss), 
                    "Entr":float(avg_epoch_entropy), 
                    "FwdLoss":float(avg_epoch_fwd_loss), 
                    "InvLoss":float(avg_epoch_inv_loss), 
                    "MaxAdv":float(self.max_adv),
                    "MinAdv":float(self.min_adv), 
                    "MinSurr":float(self.min_minsurr), 
                    "MaxSurr":float(self.max_minsurr),
                    "MaxRew":float(self.max_rew),
                    "MinRew":float(self.min_rew)} 
        self.max_adv, self.min_adv, = -1, 1
        self.max_minsurr, self.min_minsurr = -1e10, 1e10
        self.max_rew, self.min_rew = -1e15, 1e15

    def cache_losses(self, states, next_states, actions, hs, next_hs):
        """
        Creates a loss term from historical data to avoid forgetting
        within the forward dynamics model.

        states - torch FloatTensor
            minibatch of states with shape (batch_size, C, H, W)
        next_states - torch FloatTensor
            minibatch of next states with shape (batch_size, C, H, W)
        actions - torch LongTensor or FloatTensor
            minibatch of empirical actions with shape (batch_size,)
        hs - torch FloatTensor or None (B,H)
            minibatch of recurrent state vectors at time t 
        hs - torch FloatTensor or None (B,H)
            minibatch of recurrent state vectors at t+1
        """
        
        embs = self.fwd_embedder(states, hs)
        fwd_targs = self.fwd_embedder(next_states, next_hs)

        # Inverse Dynamics Loss
        if self.inv_net is not None:
            inv_preds = self.inv_net(torch.cat([embs, fwd_targs], dim=-1))
            inv_loss = F.cross_entropy(inv_preds, Variable(cuda_if(actions)))
        else:
            inv_loss = Variable(cuda_if(torch.zeros(1)))

        # Reconstruction Loss
        if self.recon_net is not None:
            recons = self.recon_net(embs)
            recon_loss = F.mse_loss(recons, states)
        else:
            recon_loss = cuda_if(torch.zeros(1))

        embs.detach(), fwd_targs.detach()
        if self.hyps['discrete_env']:
            fwd_actions = self.one_hot_encode(actions, self.net.output_space)
        else:
            fwd_actions = actions
        fwd_inputs = torch.cat([embs.data, cuda_if(fwd_actions)], dim=-1)
        if self.hyps['is_recurrent']:
            fwd_preds = self.fwd_net(fwd_inputs, hs.data)
        else:
            fwd_preds = self.fwd_net(fwd_inputs)
        fwd_loss = F.mse_loss(fwd_preds, fwd_targs.data)
        return fwd_loss, inv_loss+recon_loss

    def ppo_losses(self, states, next_states, actions, advs,
                                                       rets,
                                                       hs,
                                                       next_hs,
                                                       old_vals,
                                                       old_raw_pis):
        """
        Completes the ppo specific loss approach

        states - torch FloatTensor
            minibatch of states with shape (batch_size, C, H, W)
        next_states - torch FloatTensor
            minibatch of next states with shape (batch_size, C, H, W)
        actions - torch LongTensor
            minibatch of empirical actions with shape (batch_size,)
        advs - torch FloatTensor 
            minibatch of empirical advantages with shape (batch_size,)
        rets - torch FloatTensor
            minibatch of empirical returns with shape (batch_size,)
        hs - torch FloatTensor
            minibatch of recurrent states (batch_size,)
        next_hs - torch FloatTensor
            minibatch of the next recurrent states (batch_size,)
        old_vals: torch FloatTensor (B,)
            the old network's value predictions
        old_raw_pis: torch FloatTensor (B,A)
            the old network's policy predictions

        Returns:
            policy_loss - the PPO CLIP policy gradient shape (1,)
            val_loss - the critic loss shape (1,)
            entropy - the entropy of the action predictions shape (1,)
            fwd_loss 
        """
        hyps = self.hyps

        # Get Outputs
        if hyps['is_recurrent']:
            vals,raw_pis,_ = self.net(states,hs)
        else:
            vals, raw_pis = self.net(Variable(states))

        if hyps['norm_batch_advs']:
            advs = (advs - advs.mean())
            advs = advs / (advs.std() + 1e-7)
        self.max_adv = max(torch.max(advs), self.max_adv) # Tracking variable
        self.min_adv = min(torch.min(advs), self.min_adv) # Tracking variable
        advs = Variable(advs)

        if self.hyps['discrete_env']:
            probs = F.softmax(raw_pis, dim=-1)
            pis = probs[cuda_if(torch.arange(0,len(probs)).long()), actions]
            old_vals.detach(), old_raw_pis.detach()
            old_probs = F.softmax(old_raw_pis, dim=-1)
            old_pis = old_probs[cuda_if(torch.arange(0,len(old_probs))).long(), actions]
            ratio = pis/(old_pis+1e-5)

            # Entropy Loss
            softlogs = F.log_softmax(raw_pis, dim=-1)
            entropy_step = torch.sum(softlogs*probs, dim=-1)
            entropy = -hyps['entr_coef'] * torch.mean(entropy_step)
        else:
            mus,sigs = raw_pis
            log_ps = self.calc_log_ps(mus,sigs,actions)
            old_mus, old_sigs = old_raw_pis
            old_log_ps = self.calc_log_ps(old_mus,old_sigs,actions)
            ratio = torch.exp(log_ps-old_log_ps)

            # Entropy Loss
            entr = (torch.log(2*float(np.pi)*sigs**2+0.0001)+1)/2
            entropy = hyps['entr_coef']*entr.mean()
            entropy -= hyps['sigma_l2']*torch.norm(sigs,2).mean()

        # Policy Loss
        if len(ratio.shape) > len(advs.shape):
            advs = advs[...,None]
        surrogate1 = ratio*advs
        surrogate2 = torch.clamp(ratio, 1.-hyps['epsilon'], 1.+hyps['epsilon'])*advs
        min_surr = torch.min(surrogate1, surrogate2)
        self.max_minsurr = max(torch.max(min_surr.data), self.max_minsurr)
        self.min_minsurr = min(torch.min(min_surr.data), self.min_minsurr)
        policy_loss = -hyps['pi_coef']*min_surr.mean()

        # Value loss
        if hyps['use_gae']:
            rets = Variable(rets)
            if hyps['clip_vals']:
                clipped_vals = old_vals + torch.clamp(vals-old_vals, -hyps['epsilon'], hyps['epsilon'])
                v1 = .5*(vals.squeeze()-rets)**2
                v2 = .5*(clipped_vals.squeeze()-rets)**2
                val_loss = hyps['val_coef'] * torch.max(v1,v2).mean()
            else:
                val_loss = hyps['val_coef']*F.mse_loss(vals.squeeze(), rets)
        else:
            val_loss = Variable(cuda_if(torch.zeros(1)))


        embs = self.fwd_embedder(states, hs)
        next_embs = self.fwd_embedder(next_states, next_hs)
        # Inv Dynamics Loss
        if self.inv_net is not None:
            inv_inputs = torch.cat([embs, next_embs], dim=-1) # Backprop into embeddings
            inv_preds = self.inv_net(inv_inputs)
            inv_loss = F.cross_entropy(inv_preds, Variable(cuda_if(actions)))
        else:
            inv_loss = Variable(cuda_if(torch.zeros(1)))
        # Reconstruction Loss
        if self.recon_net is not None:
            recons = self.recon_net(embs)
            recon_loss = F.mse_loss(recons, states.data)
            inv_loss += recon_loss

        # Fwd Dynamics Loss
        if self.hyps['discrete_env']:
            fwd_actions = self.one_hot_encode(actions, self.net.output_space)
        else:
            fwd_actions = actions
        fwd_inputs = torch.cat([embs.data, cuda_if(fwd_actions)], dim=-1)
        if hyps['is_recurrent']:
            fwd_preds = self.fwd_net(fwd_inputs,hs.data)
        else:
            fwd_preds = self.fwd_net(Variable(fwd_inputs))
        fwd_loss = F.mse_loss(fwd_preds, Variable(next_embs.data))

        return policy_loss, val_loss, entropy, fwd_loss, inv_loss

    def calc_log_ps(self, mus, sigmas, actions):
        """
        calculates the log probability of pi using the mean and stddev

        mus: FloatTensor (...,)
        sigmas: FloatTensor (...,)
        actions: FloatTensor (...,)
        """
        # log_ps should be -(mu-act)^2/(2sig^2)+ln(1/(sqrt(2pi)sig))
        log_ps = -F.mse_loss(mus,actions.cuda(),reduction="none")
        log_ps = log_ps/(2*torch.clamp(sigmas**2,min=1e-4))
        logsigs = torch.log(torch.sqrt(2*float(np.pi)*sigmas))
        return log_ps - logsigs

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

    def make_advs_and_rets(self, states, next_states, rewards,
                                                      dones,
                                                      hs,
                                                      next_hs):
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
        if self.hyps['is_recurrent']:
            vals, raw_pis,_ = self.net(Variable(states), hs)
            next_vals, _,_ = self.net(Variable(next_states), next_hs)
        else:
            vals, raw_pis = self.net(Variable(states))
            next_vals, _ = self.net(Variable(next_states))
        self.net.req_grads(True)

        # Make Advantages
        advantages = self.gae(rewards.squeeze(),vals.data.squeeze(),
                                                next_vals.data.squeeze(),
                                                dones.squeeze(),
                                                self.gamma,
                                                self.lambda_)

        # Make Returns
        if self.use_nstep_rets: 
            returns = advantages + vals.data.squeeze()
        else: 
            # Include bootstrap
            rewards[dones==1] = rewards[dones==1] + self.hyps['gamma']*next_vals.data.squeeze()[dones==1] 
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

    def update_cache(self, shared_data, keys):
        if self.cache is None:
            self.cache = dict()
            for key in keys:
                self.cache[key] = shared_data[key].clone()
        elif self.hyps['cache_size'] > 0:
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

    def save_model(self, net_file_name, fwd_save_file, optim_file,
                                                 fwd_optim_file,
                                                 inv_save_file=None,
                                                 recon_save_file=None,
                                                 reconinv_optim_file=None):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        torch.save(self.fwd_net.state_dict(), fwd_save_file)
        if optim_file is not None:
            torch.save(self.optim.state_dict(), optim_file)
            torch.save(self.fwd_optim.state_dict(), fwd_optim_file)
            save_optim = False
            if inv_save_file is not None and self.inv_net is not None:
                torch.save(self.inv_net.state_dict(), inv_save_file)
                save_optim = True
            if recon_save_file is not None and self.recon_net is not None:
                torch.save(self.recon_net.state_dict(), recon_save_file)
                save_optim = True
            if save_optim and reconinv_optim_file is not None:
                torch.save(self.reconinv_optim.state_dict(),
                                        reconinv_optim_file)
    
    def new_lr(self, new_lr):
        optim_type = self.hyps['optim_type']
        new_optim = self.new_optim(self.net.parameters(), new_lr, optim_type)
        new_optim.load_state_dict(self.optim.state_dict())
        self.optim = new_optim

    def new_fwd_lr(self, new_lr):
        optim_type = self.hyps['fwd_optim_type']
        new_optim = self.new_optim(self.fwd_net.parameters(),
                                          new_lr, optim_type)
        new_optim.load_state_dict(self.optim.state_dict())
        self.fwd_optim = new_optim

    def new_optim(self, params, lr, optim_type):
        if optim_type == 'rmsprop':
            new_optim = optim.RMSprop(params, lr=lr) 
        elif optim_type == 'adam':
            new_optim = optim.Adam(params, lr=lr) 
        else:
            new_optim = optim.RMSprop(params, lr=lr) 
        return new_optim

    #def fwd_bptt(self, datas):
    #    """
    #    Func used to backprop through the recurrence in the forward
    #    dynamics model.
    #    """
    #    hyps = self.hyps
    #    states = datas['states']
    #    actions = datas['actions']
    #    dones = datas['dones']

    #    og_len = len(states)
    #    n_rollouts, n_tsteps = hyps['n_rollouts'], hyps['n_tsteps']
    #    states = states.reshape(n_rollouts,n_tsteps,*states.shape[1:])
    #    if len(actions.shape) == 1:
    #        actions = actions.reshape(n_rollouts, n_tsteps)
    #    else:
    #        actions = actions.reshape(n_rollouts, n_tsteps,
    #                                              *actions.shape[1:])
    #    dones = dones.reshape(n_rollouts, n_tsteps)

    #    # TODO: Collect reliable starting hs
    #    # (beware of n_envs vs n_rollouts problem)
    #    # TODO: make fresh h func
    #    h_init = self.fwd_net.fresh_h(n_rollouts)
    #    h = self.prev_fwd_h

    #    for step in 












