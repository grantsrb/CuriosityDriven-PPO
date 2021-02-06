import os
import sys
import gym
from logger import Logger
from runner import Runner, StatsRunner
from updater import Updater
from environments import SeqEnv
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gc
import resource
import torch.multiprocessing as mp
import copy
import time
from collections import deque
from utils import cuda_if, deque_maxmin
from ml_utils.utils import try_key
from models.embedder import CatModule, Ensemble, FwdRunModel,RecurrentFwdModel, ContrastModel
from transformer.models import Attncoder

class CurioPPO:
    def __init__(self):
        pass

    def train(self, hyps): 
        """
        hyps - dictionary of required hyperparameters
            type: dict
        """

        # Initial settings
        if "randomizeObjs" in hyps:
            assert False, "you mean randomizeObs, not randomizeObjs"
        if "audibleTargs" in hyps and hyps['audibleTargs'] > 0:
            hyps['aud_targs'] = True
            if verbose: print("Using audible targs!")
        countOut = try_key(hyps, 'countOut', 0)
        if countOut and not hyps['endAtOrigin']:
            assert False, "endAtOrigin must be true for countOut setting"

        # Print Hyperparameters To Screen
        items = list(hyps.items())
        for k, v in sorted(items):
            print(k+":", v)

        # Make Save Files
        if "save_folder" in hyps:
            save_folder = hyps['save_folder']
        else:
            save_folder = "./saved_data/"

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        base_name = save_folder + hyps['exp_name']
        net_save_file = base_name+"_net.p"
        fwd_save_file = base_name+"_fwd.p"
        best_net_file = base_name+"_best.p"
        optim_save_file = base_name+"_optim.p"
        fwd_optim_file = base_name+"_fwdoptim.p"
        hyps['fwd_emb_file'] = base_name+"_fwdemb.p"
        if hyps['inv_model'] is not None:
            inv_save_file = base_name+"_invnet.p"
            reconinv_optim_file = base_name+"_reconinvoptim.p"
        else:
            inv_save_file = None
            reconinv_optim_file = None
        if hyps['recon_model'] is not None:
            recon_save_file = base_name+"_reconnet.p"
            reconinv_optim_file = base_name+"_reconinvoptim.p"
        else: recon_save_file = None
        if hyps['contrast']: contr_save_file = base_name+"_contrnet.p"
        else:
            contr_save_file = None
            reconinv_optim_file = base_name+"_reconinvoptim.p"
        log_file = base_name+"_log.txt"
        if hyps['resume']: log = open(log_file, 'a')
        else: log = open(log_file, 'w')
        for k, v in sorted(items):
            log.write(k+":"+str(v)+"\n")

        # Miscellaneous Variable Prep
        logger = Logger()
        shared_len = hyps['n_tsteps']*hyps['n_rollouts']
        float_params = dict()
        if "float_params" not in hyps:
            try:
                keys = hyps['game_keys']
                hyps['float_params'] = {k:try_key(hyps,k,0) for k in keys}
                if "minObjLoc" not in hyps:
                    hyps['float_params']["minObjLoc"] = 0.27
                    hyps['float_params']["maxObjLoc"] = 0.73
                float_params = hyps['float_params']
            except: pass
        env = SeqEnv(hyps['env_type'], hyps['seed'],
                                            worker_id=None,
                                            float_params=float_params)
        hyps['discrete_env'] = hasattr(env.action_space, "n")
        obs = env.reset()
        prepped = hyps['preprocess'](obs)
        hyps['state_shape'] = [hyps['n_frame_stack']*prepped.shape[0],
                              *prepped.shape[1:]]
        if not hyps['discrete_env']:
            action_size = int(np.prod(env.action_space.shape))
        elif hyps['env_type'] == "Pong-v0":
            action_size = 3
        else:
            action_size = env.action_space.n
        hyps['action_shift'] = (4-action_size)*(hyps['env_type']=="Pong-v0") 
        print("Obs Shape:,",obs.shape)
        print("Prep Shape:,",prepped.shape)
        print("State Shape:,",hyps['state_shape'])
        print("Num Samples Per Update:", shared_len)
        if not (hyps['n_cache_refresh'] <= shared_len or hyps['cache_size'] == 0):
            hyps['n_cache_refresh'] = shared_len
        print("Samples Wasted in Update:", shared_len % hyps['batch_size'])
        try: env.close()
        except: pass
        del env

        # Prepare Shared Variables
        shared_data = {
            'states': torch.zeros(shared_len,
                                *hyps['state_shape']).share_memory_(),
            'next_states': torch.zeros(shared_len,
                                *hyps['state_shape']).share_memory_(),
            'dones':torch.zeros(shared_len).share_memory_(),
            'rews':torch.zeros(shared_len).share_memory_(),
            'hs':torch.zeros(shared_len,hyps['h_size']).share_memory_(),
            'next_hs':torch.zeros(shared_len,hyps['h_size']).share_memory_(),
            'fwd_hs':torch.zeros(shared_len,hyps['h_size']).share_memory_()}
        if hyps['discrete_env']:
            shared_data['actions'] = torch.zeros(shared_len).long().share_memory_()
        else:
            shape = (shared_len, action_size)
            shared_data['actions']=torch.zeros(shape).float().share_memory_()
        shared_data = {k: cuda_if(v) for k,v in shared_data.items()}
        n_rollouts = hyps['n_rollouts']
        gate_q = mp.Queue(n_rollouts)
        stop_q = mp.Queue(n_rollouts)
        end_q = mp.Queue(1)
        reward_q = mp.Queue(1)
        reward_q.put(-1)

        # Make Runners
        runners = []
        for i in range(hyps['n_envs']):
            runner = Runner(shared_data, hyps, gate_q, stop_q,
                                                       end_q,
                                                       reward_q)
            runners.append(runner)

        # Make the Networks
        h_size = hyps['h_size']
        net = hyps['model'](hyps['state_shape'], action_size, h_size,
                                            bnorm=hyps['use_bnorm'],
                                            lnorm=hyps['use_lnorm'],
                                            discrete_env=hyps['discrete_env'])
        # Fwd Dynamics
        hyps['is_recurrent'] = hasattr(net, "fresh_h")
        if hyps['recurrent_fwd']:
            fwd_net = RecurrentFwdModel(h_size=h_size,
                                        a_size=action_size,
                                        rnn_type=hyps['rnn_type'],
                                        lnorm=hyps['fwd_lnorm'])
        else:
            intl_size = h_size+action_size + hyps['is_recurrent']*h_size
            if hyps['fwd_lnorm']:
                block = [nn.LayerNorm(intl_size)]
            block = [nn.Linear(intl_size, h_size), 
                nn.ReLU(), nn.Linear(h_size, h_size), 
                nn.ReLU(), nn.Linear(h_size, h_size)]
            fwd_net = nn.Sequential(*block)

            # Allows us to argue an h vector along with embedding to
            # forward func
            if hyps['is_recurrent']:
                fwd_net = CatModule(fwd_net) 
            if hyps['ensemble']:
                fwd_net = Ensemble(fwd_net)
        fwd_net = cuda_if(fwd_net)

        # Inverse Dynamics
        if hyps['inv_model'] is not None:
            inv_net = hyps['inv_model'](h_size, action_size)
            inv_net = cuda_if(inv_net)
        else:
            inv_net = None

        # Reconstruction Model
        if hyps['recon_model'] is not None:
            recon_net = hyps['recon_model'](emb_size=h_size,
                                       img_shape=hyps['state_shape'],
                                       fwd_bnorm=hyps['fwd_bnorm'],
                                       deconv_ksizes=hyps['recon_ksizes'])
            recon_net = cuda_if(recon_net)
        else:
            recon_net = None

        # Contrastive Model
        if hyps['contrast']:
            args = {**hyps}
            args['emb_size'] = h_size
            args['n_layers'] = hyps['dec_layers']
            args['attn_size'] = 64
            args['n_heads'] = 8
            args['out_size'] = 2
            contr_net = ContrastModel(**args)
            contr_net = cuda_if(contr_net)
        else:
            contr_net = None

        if hyps['resume']:
            net.load_state_dict(torch.load(net_save_file))
        base_net = copy.deepcopy(net)
        net = cuda_if(net)
        net.share_memory()
        base_net = cuda_if(base_net)
        hyps['is_recurrent'] = hasattr(net, "fresh_h")

        fwd_embedder = net.embedder
        if hyps['seperate_embs']:
            if "fwd_emb_model" in hyps and hyps['fwd_emb_model'] is not None:
                args = {**hyps}
                args['bnorm'] = hyps['fwd_bnorm']
                args['lnorm'] = hyps['fwd_lnorm']
                args['input_space'] = args['state_shape']
                args['emb_size'] = hyps['h_size']
                args['img_shape'] = hyps['state_shape']
                fwd_embedder = cuda_if(hyps['fwd_emb_model'](**args))
            else:
                fwd_embedder = copy.deepcopy(net.embedder)
            if hyps['resume']:
                f = hyps['fwd_emb_file']
                fwd_embedder.load_state_dict(torch.load(f))

        if hyps['recurrent_fwd']:
            if hyps['ensemble']: raise NotImplementedError
            # Fwd net is an instance of RecurrentFwdModel or similar
            fwd_net = FwdRunModel(fwd_embedder, fwd_net)

        if hyps['resume']:
            fwd_net.load_state_dict(torch.load(fwd_save_file))
            if inv_net is not None:
                inv_net.load_state_dict(torch.load(inv_save_file))
            if recon_net is not None:
                recon_net.load_state_dict(torch.load(recon_save_file))
            if contr_net is not None:
                recon_net.load_state_dict(torch.load(contr_save_file))

        # Start Data Collection
        print("Making New Processes")
        procs = []
        args = (net,fwd_net)
        for i in range(len(runners)):
            proc = mp.Process(target=runners[i].run, args=args)
            procs.append(proc)
            proc.start()
            print(i, "/", len(runners), end='\r')
        for i in range(n_rollouts):
            gate_q.put(i)

        # Make Updater
        updater = Updater(base_net, fwd_net, hyps, fwd_embedder,
                                                   inv_net,
                                                   recon_net,
                                                   contr_net)
        if hyps['resume']:
            updater.optim.load_state_dict(torch.load(optim_save_file))
            updater.fwd_optim.load_state_dict(torch.load(fwd_optim_file))
            if inv_net is not None:
                updater.reconinv_optim.load_state_dict(torch.load(reconinv_optim_file))
        updater.optim.zero_grad()
        updater.net.train(mode=True)
        updater.net.req_grads(True)

        # Prepare Decay Precursors
        entr_coef_diff = hyps['entr_coef'] - hyps['entr_coef_low']
        epsilon_diff = hyps['epsilon'] - hyps['epsilon_low']
        lr_diff = hyps['lr'] - hyps['lr_low']
        gamma_diff = hyps['gamma_high'] - hyps['gamma']

        # Training Loop
        past_rews = deque([0]*hyps['n_past_rews'])
        last_avg_rew = 0
        best_rew_diff = 0
        best_avg_rew = -10000
        best_eval_rew = -10000
        ep_eval_rew = 0
        eval_rew = 0

        epoch = 0
        done_count = 0
        T = 0
        try:
            while T < hyps['max_tsteps']:
                basetime = time.time()
                epoch += 1

                # Collect data
                for i in range(n_rollouts):
                    stop_q.get()
                T += shared_len

                data_copy = {}
                for k,v in shared_data.items():
                    if hasattr(v,"clone"): data_copy[k] = v.clone()
                    else:
                        print("non tensor k", k, v) 
                        data_copy[k] = v

                # Resume Data Collection
                for i in range(n_rollouts):
                    gate_q.put(i)


                # Reward Stats
                avg_reward = reward_q.get()
                reward_q.put(avg_reward)
                last_avg_rew = avg_reward
                done_count += data_copy['dones'].sum().item()
                new_best = False
                if avg_reward > best_avg_rew and done_count > n_rollouts:
                    new_best = True
                    best_avg_rew = avg_reward
                    updater.save_model(best_net_file, fwd_save_file,
                                                         None, None)
                eval_rew = data_copy['rews'].mean()
                if eval_rew > best_eval_rew:
                    best_eval_rew = eval_rew
                    save_names = [net_save_file, fwd_save_file,
                                                 optim_save_file,
                                                 fwd_optim_file,
                                                 inv_save_file,
                                                 recon_save_file,
                                                 contr_save_file,
                                                 reconinv_optim_file]

                    for i in range(len(save_names)):
                        if save_names[i] is not None:
                            splt = save_names[i].split(".")
                            splt[0] = splt[0]+"_best"
                            save_names[i] = ".".join(splt)
                    updater.save_model(*save_names)
                s = "EvalRew: {:.5f} | BestEvalRew: {:.5f}"
                print(s.format(eval_rew, best_eval_rew))

                # Calculate the Loss and Update nets
                updater.update_model(data_copy)
                net.load_state_dict(updater.net.state_dict()) # update all collector nets
                
                # Decay HyperParameters
                if hyps['decay_eps']:
                    updater.epsilon = (1-T/(hyps['max_tsteps']))*epsilon_diff + hyps['epsilon_low']
                    print("New Eps:", updater.epsilon)
                if hyps['decay_lr']:
                    new_lr = (1-T/(hyps['max_tsteps']))*lr_diff + hyps['lr_low']
                    updater.new_lr(new_lr)
                    print("New lr:", new_lr)
                if hyps['decay_entr']:
                    updater.entr_coef = entr_coef_diff*(1-T/(hyps['max_tsteps']))+hyps['entr_coef_low']
                    print("New Entr:", updater.entr_coef)
                if hyps['incr_gamma']:
                    updater.gamma = gamma_diff*(T/(hyps['max_tsteps']))+hyps['gamma']
                    print("New Gamma:", updater.gamma)

                # Periodically save model
                if epoch % 10 == 0 or epoch == 1:
                    updater.save_model(net_save_file, fwd_save_file,
                                                      optim_save_file,
                                                      fwd_optim_file,
                                                      inv_save_file,
                                                      recon_save_file,
                                                      contr_save_file,
                                                      reconinv_optim_file)

                # Print Epoch Data
                past_rews.popleft()
                past_rews.append(avg_reward)
                max_rew, min_rew = deque_maxmin(past_rews)
                print("Epoch", epoch, "– T =", T, "-- Folder:", base_name)
                if not hyps['discrete_env']:
                    s = ("{:.5f} | "*net.logsigs.shape[1])
                    s = s.format(*[x.item() for x in torch.exp(net.logsigs[0])])
                    print("Sigmas:", s)
                updater.print_statistics()
                avg_action = data_copy['actions'].float().mean().item()
                print("Grad Norm:",float(updater.norm),"– Avg Action:",avg_action,"– Best AvgRew:",best_avg_rew)
                print("Avg Rew:", avg_reward, "– High:", max_rew, "– Low:", min_rew, end='\n')
                updater.log_statistics(log, T, avg_reward, avg_action, best_avg_rew)
                updater.info['AvgRew'] = avg_reward
                updater.info['EvalRew'] = eval_rew
                logger.append(updater.info, x_val=T)

                # Check for memory leaks
                gc.collect()
                max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print("Time:", time.time()-basetime)
                if 'hyp_search_count' in hyps and hyps['hyp_search_count'] > 0 and hyps['search_id'] != None:
                    print("Search:", hyps['search_id'], "/", hyps['hyp_search_count'])
                print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))
                if updater.info["VLoss"] == float('inf') or updater.norm == float('inf'):
                    break
        except KeyboardInterrupt:
            pass

        end_q.put(1)
        time.sleep(1)
        logger.make_plots(base_name)
        log.write("\nBestRew:"+str(best_avg_rew))
        log.close()
        # Close processes
        for p in procs:
            p.terminate()
        return best_avg_rew
