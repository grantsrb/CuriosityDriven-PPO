from hyperparams import HyperParams, make_hyper_range, hyper_search
from curio_ppo import CurioPPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = CurioPPO()
    hyps = dict()
    hyp_ranges = {
                "inv_lr": [1e-4, 1e-5, 1e-6],
                "fwd_lr": [1e-4, 1e-5],
                }
    keys = list(hyp_ranges.keys())
    hyps['use_idf'] = True
    hyps['lr'] = .0001
    hyps['val_coef'] = .005
    hyps['norm_rews'] = True
    hyps['use_nstep_rets'] = True
    hyps['clip_vals'] = False
    hyps['lambda_'] = .95
    hyps['gamma'] = .99
    hyps['entr_coef'] = .008
    hyps['dyn_coef'] = .5
    hyps['cache_coef'] = .5
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "invdyn"
    hyps['use_gae'] = True
    hyps['n_tsteps'] = 128
    hyps['n_rollouts'] = 12
    hyps['n_envs'] = 12
    hyps['max_tsteps'] = 6000000
    hyps['n_frame_stack'] = 3
    hyps['optim_type'] = 'rmsprop'
    hyps['cache_size'] = 2000
    hyps['n_cache_refresh'] = 200
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, ppo_trainer, search_log)
    search_log.close()

