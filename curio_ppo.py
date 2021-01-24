import os
import sys
import gym
from logger import Logger
from runner import Runner
from updater import Updater
import torch
from torch.autograd import Variable
import numpy as np
import gc
import resource
import torch.multiprocessing as mp
import copy
import time
from collections import deque
from utils import cuda_if, deque_maxmin

class CurioPPO:
    def __init__(self):
        pass

    def train(self, hyps): 
        """
        hyps - dictionary of required hyperparameters
            type: dict
        """

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
        best_net_file = base_name+"_best.p"
        optim_save_file = base_name+"_optim.p"
        fwd_optim_file = base_name+"_fwdoptim.p"
        hyps['fwd_emb_file'] = base_name+"_fwdemb.p"
        if hyps['inv_model'] is not None:
            inv_save_file = base_name+"_invnet.p"
            inv_optim_file = base_name+"_invoptim.p"
        else:
            inv_save_file = None
            inv_optim_file = None
        log_file = base_name+"_log.txt"
        if hyps['resume']: log = open(log_file, 'a')
        else: log = open(log_file, 'w')
        for k, v in sorted(items):
            log.write(k+":"+str(v)+"\n")

        # Miscellaneous Variable Prep
        logger = Logger()
        shared_len = hyps['n_tsteps']*hyps['n_rollouts']
        env = gym.make(hyps['env_type'])
        hyps['discrete_env'] = hasattr(env.action_space, "n")
        obs = env.reset()
        prepped = hyps['preprocess'](obs)
        hyps['state_shape'] = [hyps['n_frame_stack']] + [*prepped.shape[1:]]
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
        assert hyps['n_cache_refresh'] <= shared_len or hyps['cache_size'] == 0
        print("Samples Wasted in Update:", shared_len % hyps['batch_size'])
        del env

        # Prepare Shared Variables
        shared_data = {'states': cuda_if(torch.zeros(shared_len, *hyps['state_shape']).share_memory_()),
                'next_states': cuda_if(torch.zeros(shared_len, *hyps['state_shape']).share_memory_()),
                'dones': cuda_if(torch.zeros(shared_len).share_memory_())}
        if hyps['discrete_env']:
            shared_data['actions'] = cuda_if(torch.zeros(shared_len).long().share_memory_())
        else:
            shape = (shared_len, action_size)
            shared_data['actions'] = cuda_if(torch.zeros(shape).float().share_memory_())
        n_rollouts = hyps['n_rollouts']
        gate_q = mp.Queue(n_rollouts)
        stop_q = mp.Queue(n_rollouts)
        reward_q = mp.Queue(1)
        reward_q.put(-1)

        # Make Runners
        runners = []
        for i in range(hyps['n_envs']):
            runner = Runner(shared_data, hyps, gate_q, stop_q, reward_q)
            runners.append(runner)

        # Make Network
        h_size = hyps['h_size']
        net = hyps['model'](hyps['state_shape'], action_size, h_size,
                                            bnorm=hyps['use_bnorm'],
                                            discrete_env=hyps['discrete_env'])
        if hyps['inv_model'] is not None:
            inv_net = hyps['inv_model'](h_size, action_size)
            inv_net = cuda_if(inv_net)
        else:
            inv_net = None
        if hyps['resume']:
            net.load_state_dict(torch.load(net_save_file))
            if inv_net is not None:
                inv_net.load_state_dict(torch.load(inv_save_file))
        base_net = copy.deepcopy(net)
        net = cuda_if(net)
        net.share_memory()
        base_net = cuda_if(base_net)

        # Start Data Collection
        print("Making New Processes")
        procs = []
        for i in range(len(runners)):
            proc = mp.Process(target=runners[i].run, args=(net,))
            procs.append(proc)
            proc.start()
            print(i, "/", len(runners), end='\r')
        for i in range(n_rollouts):
            gate_q.put(i)

        # Make Updater
        updater = Updater(base_net, hyps, inv_net)
        if hyps['resume']:
            updater.optim.load_state_dict(torch.load(optim_save_file))
            updater.fwd_optim.load_state_dict(torch.load(fwd_optim_file))
            if inv_net is not None:
                updater.inv_optim.load_state_dict(torch.load(inv_optim_file))
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
        best_avg_rew = -100
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

                # Reward Stats
                avg_reward = reward_q.get()
                reward_q.put(avg_reward)
                last_avg_rew = avg_reward
                done_count += shared_data['dones'].sum().item()
                if avg_reward > best_avg_rew and done_count > n_rollouts:
                    best_avg_rew = avg_reward
                    updater.save_model(best_net_file, None, None)

                # Calculate the Loss and Update nets
                updater.update_model(shared_data)
                net.load_state_dict(updater.net.state_dict()) # update all collector nets
                
                # Resume Data Collection
                for i in range(n_rollouts):
                    gate_q.put(i)

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
                if epoch % 10 == 0:
                    updater.save_model(net_save_file, optim_save_file, fwd_optim_file, inv_save_file, inv_optim_file)

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
                avg_action = shared_data['actions'].float().mean().item()
                print("Grad Norm:",float(updater.norm),"– Avg Action:",avg_action,"– Best AvgRew:",best_avg_rew)
                print("Avg Rew:", avg_reward, "– High:", max_rew, "– Low:", min_rew, end='\n')
                updater.log_statistics(log, T, avg_reward, avg_action, best_avg_rew)
                updater.info['AvgRew'] = avg_reward
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

        logger.make_plots(base_name)
        log.write("\nBestRew:"+str(best_avg_rew))
        log.close()
        # Close processes
        for p in procs:
            p.terminate()
        return best_avg_rew
