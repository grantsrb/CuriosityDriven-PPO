from hyperparams import HyperParams, hyper_search, make_hyper_range
from curio_ppo import CurioPPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = CurioPPO()
    hyps = dict()
    hyp_ranges = {
                "lr": [2.5e-5, 5e-5, 7.5e-5],
                "dyn_coef": [.5, .65, .85],
                }
    keys = list(hyp_ranges.keys())
    hyps['lambda_'] = .95
    hyps['gamma'] = .99
    hyps['entr_coef'] = .008
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "brkout2"
    hyps['n_tsteps'] = 128
    hyps['n_rollouts'] = 11
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 8000000
    hyps['n_frame_stack'] = 3
    hyps['optim_type'] = 'rmsprop'
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, ppo_trainer, search_log)
    search_log.close()

