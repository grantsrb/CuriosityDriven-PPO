from hyperparams import HyperParams
from curio_ppo import CurioPPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = CurioPPO()
    hyper_params = HyperParams()
    ppo_trainer.train(hyper_params.hyps)

