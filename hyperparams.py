import sys
import preprocessing
from models import ConvModel, FCModel, A3CModel, InvDynamics
import numpy as np

class HyperParams:
    def __init__(self, arg_hyps=None):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"locgame",
                    "seed": 121314,
                    "model_type":"conv", # Options include 'dense', 'conv', 'a3c'
                    "env_type":"~/loc_games/LocationGame2dLinux_8/LocationGame2dLinux.x86_64", 
                    "optim_type":'rmsprop', # Options: rmsprop, adam
                    "fwd_optim_type":'rmsprop', # Options: rmsprop, adam
                    "inv_optim_type":'adam', # Options: rmsprop, adam
                    "save_folder":"/media/grantsrb/curioppo_saves/"
                    }

        hyp_dict['int_hyps'] = {
                    "n_epochs": 3, # PPO update epoch count
                    "batch_size": 128, # PPO update batch size
                    "h_size": 256,
                    "cache_batch": 128, # Batch size for cached data in forward dynamics loss
                    "max_tsteps": int(4e7),
                    "n_tsteps": 64, # Maximum number of tsteps per rollout per perturbed copy
                    "n_envs": 6, # Number of parallel python processes
                    "n_frame_stack":2,# Number of frames to stack in MDP state
                    "n_rollouts": 12,
                    "n_past_rews":25,
                    "cache_size":2000,
                    "n_cache_refresh":200,
                    "grid_size":15,
                    "unit_size":4,
                    "n_foods":2,

                    "validation":0,
                    "visibleOrigin":1,
                    "endAtOrigin":1,
                    "egoCentered":0,
                    "absoluteCoords":1,
                    "smoothMovement":0,
                    "restrictCamera":1,
                    "randomizeObs":0,
                    "specGoalObjs":0,
                    "randObjOrder":0,
                    "visibleTargs":0,
                    "audibleTargs":0,
                    "countOut":1,
                    "visibleCount":1,
                    "deleteTargets":1,
                    "meritForward":1,
                    "minObjCount":2,
                    "maxObjCount":5,
                    }

        hyp_dict['float_hyps'] = {
                    "fwd_lr":0.00001,
                    "inv_lr":0.0001,
                    "lr":0.0001,
                    "lr_low": float(1e-12),
                    "lambda_":.95,
                    "gamma":.99,
                    "gamma_high":.995,
                    "pi_coef":1,
                    "val_coef":.005,
                    "entr_coef":0.008,
                    "entr_coef_low":.001,
                    "sigma_l2":0,
                    "max_norm":.5,
                    "epsilon": .2, # PPO update clipping constant
                    "epsilon_low":.05,
                    "fwd_coef":.5,# Scaling factor for fwd dynamics portion of loss. range: 0-1
                    "inv_coef":.5, # Scaling factor for inverse dynamics portion of loss. range: 0-1
                    'cache_coef': .5, # Portion of inverse and forward dynamics losses from cached data. range: 0-1
                    "minObjLoc":0.27,
                    "maxObjLoc":0.73,
                    }

        hyp_dict['bool_hyps'] = {
                    "resume":False,
                    "render": False, # Do not use in training scheme
                    "clip_vals": False,
                    "decay_eps": False,
                    "decay_lr": False,
                    "decay_entr": False,
                    "incr_gamma": False,
                    "use_nstep_rets": True,
                    "norm_advs": True,
                    "norm_batch_advs": False,
                    "use_bnorm": False,
                    "use_gae": True,
                    "norm_rews": True,
                    "running_rew_norm": False,
                    "use_idf": False, # IDF stands for Inverse Dynamics Features
                    "seperate_embs": True, # Uses seperate embedding model for policy and dynamics, gradients are not backpropagated in either case
                    }
        hyp_dict["list_hyps"] = {
                    "game_keys":["validation", "visibleOrigin", "endAtOrigin",
                        "egoCentered", "absoluteCoords", "smoothMovement",
                        "restrictCamera", "randomizeObs", "specGoalObjs",
                        "randObjOrder", "visibleTargs",
                        "audibleTargs", "minObjLoc", "maxObjLoc",
                        "minObjCount", "maxObjCount", "countOut",
                        "visibleCount", "deleteTargets", "meritForward"
                        ],
                    }
        self.hyps = self.read_command_line(hyp_dict)
        if arg_hyps is not None:
            for arg_key in arg_hyps.keys():
                self.hyps[arg_key] = arg_hyps[arg_key]

        # Hyperparameter Manipulations
        self.hyps['grid_size'] = [self.hyps['grid_size'],self.hyps['grid_size']]
        if self.hyps['batch_size'] > self.hyps['n_rollouts']*self.hyps['n_tsteps']:
            self.hyps['batch_size'] = self.hyps['n_rollouts']*self.hyps['n_tsteps']

        # Model Type
        model_type = self.hyps['model_type'].lower()
        if "conv" == model_type:
            self.hyps['model'] = ConvModel
        elif "a3c" == model_type:
            self.hyps['model'] = A3CModel
        elif "fc" == model_type or "dense" == model_type:
            self.hyps['model'] = FCModel
        else:
            self.hyps['model'] = ConvModel

        if self.hyps['use_idf']:
            self.hyps['inv_model'] = InvDynamics
        else:
            self.hyps['inv_model'] = None

        # Preprocessor Type
        env_type = self.hyps['env_type'].lower()
        if "pong" in env_type:
            self.hyps['preprocess'] = preprocessing.pong_prep
        elif "breakout" in env_type:
            self.hyps['preprocess'] = preprocessing.breakout_prep
        elif "snake" in env_type:
            self.hyps['preprocess'] = preprocessing.snake_prep
        elif "pendulum" in env_type or "mountaincar" in env_type:
            self.hyps['preprocess'] = preprocessing.pendulum_prep
        elif "loc" in env_type:
            self.hyps['preprocess'] = preprocessing.center_zero2one
        else:
            self.hyps['preprocess'] = preprocessing.null_prep

    def read_command_line(self, hyps_dict):
        """
        Reads arguments from the command line. If the parameter name is not declared in __init__
        then the command line argument is ignored.
    
        Pass command line arguments with the form parameter_name=parameter_value
    
        hyps_dict - dictionary of hyperparameter dictionaries with keys:
                    "bool_hyps" - dictionary with hyperparameters of boolean type
                    "int_hyps" - dictionary with hyperparameters of int type
                    "float_hyps" - dictionary with hyperparameters of float type
                    "string_hyps" - dictionary with hyperparameters of string type
        """
        
        bool_hyps = hyps_dict['bool_hyps']
        int_hyps = hyps_dict['int_hyps']
        float_hyps = hyps_dict['float_hyps']
        string_hyps = hyps_dict['string_hyps']
        list_hyps = hyps_dict['list_hyps']
        
        if len(sys.argv) > 1:
            for arg in sys.argv:
                arg = str(arg)
                sub_args = arg.split("=")
                if sub_args[0] in bool_hyps:
                    bool_hyps[sub_args[0]] = sub_args[1] == "True"
                elif sub_args[0] in float_hyps:
                    float_hyps[sub_args[0]] = float(sub_args[1])
                elif sub_args[0] in string_hyps:
                    string_hyps[sub_args[0]] = sub_args[1]
                elif sub_args[0] in int_hyps:
                    int_hyps[sub_args[0]] = int(sub_args[1])
    
        return {**bool_hyps, **float_hyps, **int_hyps, **string_hyps, **list_hyps}

# Methods

def hyper_search(hyps, hyp_ranges, keys, idx, trainer, search_log):
    """
    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of ranges for hyperparameters to take over the search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list of values to search over for that hyperparameter
    keys - keys of the hyperparameters to be searched over. Used to
            allow order of hyperparameter search
    idx - the index of the current key to be searched over
    trainer - trainer object that handles training of model
    """
    if idx >= len(keys):
        if 'search_id' not in hyps:
            hyps['search_id'] = 0
            hyps['exp_name'] = hyps['exp_name']+"0"
            hyps['hyp_search_count'] = np.prod([len(hyp_ranges[key]) for key in keys])
        id_ = len(str(hyps['search_id']))
        hyps['search_id'] += 1
        hyps['exp_name'] = hyps['exp_name'][:-id_]+str(hyps['search_id'])
        best_avg_rew = trainer.train(hyps)
        params = [str(key)+":"+str(hyps[key]) for key in keys]
        search_log.write(", ".join(params)+" – BestRew:"+str(best_avg_rew)+"\n")
        search_log.flush()
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_search(hyps, hyp_ranges, keys, idx+1, trainer, search_log)
    return

def make_hyper_range(low, high, range_len, method="log"):
    if method.lower() == "random":
        param_vals = np.random.random(low, high+1e-5, size=range_len)
    elif method.lower() == "uniform":
        step = (high-low)/(range_len-1)
        pos_step = (step > 0)
        range_high = high+(1e-5)*pos_step-(1e-5)*pos_step
        param_vals = np.arange(low, range_high, step=step)
    else:
        range_low = np.log(low)/np.log(10)
        range_high = np.log(high)/np.log(10)
        step = (range_high-range_low)/(range_len-1)
        arange = np.arange(range_low, range_high, step=step)
        if len(arange) < range_len:
            arange = np.append(arange, [range_high])
        param_vals = 10**arange
    param_vals = [float(param_val) for param_val in param_vals]
    return param_vals
