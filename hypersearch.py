from hyperparams import HyperParams, make_hyper_range, hyper_search
from curio_ppo import CurioPPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = CurioPPO()
    hyps = dict()
    hyp_ranges = {
                'fwd_lr': [1e-5, 1e-6, 1e-7, 1e-4],
                'lr': [1e-5, 1e-6, 1e-7, 1e-4],
                'inv_lr': [1e-6],
                }
    #keys = list(hyp_ranges.keys())
    keys = ['inv_lr', 'lr', 'fwd_lr']
    hyps['use_idf'] = False
    hyps['fwd_coef'] = .8
    hyps['val_coef'] = .5
    hyps['norm_rews'] = False
    hyps['use_nstep_rets'] = True
    hyps['clip_vals'] = False
    hyps['use_gae'] = True
    hyps['seperate_embs'] = True
    hyps['lambda_'] = .95
    hyps['gamma'] = .99
    hyps['entr_coef'] = .005
    hyps['inv_coef'] = .5 # Portion due to cache in inverse dynamics
    hyps['cache_coef'] = .5
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "4sepembs"
    hyps['n_tsteps'] = 64
    hyps['n_rollouts'] = 24
    hyps['n_envs'] = 12
    hyps['n_epochs'] = 4
    hyps['max_tsteps'] = 50000000
    hyps['n_frame_stack'] = 3
    hyps['optim_type'] = 'rmsprop'
    hyps['cache_size'] = 10000
    hyps['n_cache_refresh'] = 200
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, ppo_trainer, search_log)
    search_log.close()

