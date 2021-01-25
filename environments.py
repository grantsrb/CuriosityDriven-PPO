import numpy as np
import gym
import os
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import time

class SeqEnv:
    def __init__(self, env_name, seed=int(time.time()),
                                      worker_id=None,
                                      float_params=dict(),
                                      **kwargs):
        """
        env_name: str
            the name of the environment
        seed: int
            the random seed for the environment
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        """
        self.env_name = env_name
        self.seed = seed
        self.worker_id = worker_id
        self.float_params = float_params

        try:
            self.env = gym.make(env_name)
            self.env.seed(seed)
            self.is_gym = True
        except Exception as e:
            self.env = UnityGymEnv(env_name=self.env_name,
                                   seed=self.seed,
                                   worker_id=self.worker_id,
                                   float_params=self.float_params)
            self.is_gym = False
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class UnityGymEnv:
    def __init__(self, env_name, seed=int(time.time()),
                                 worker_id=None,
                                 float_params=dict(),
                                 **kwargs):
        """
        env_name: str
            the name of the environment
        seed: int
            the random seed for the environment
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        """
        self.env_name = env_name
        self.seed = seed
        self.worker_id = worker_id
        self.float_params = float_params

        self.env = self.make_unity_env(env_name,
                                       seed=self.seed,
                                       worker_id=self.worker_id,
                                       float_params=float_params,
                                       **kwargs)
        obs = self.reset()
        self.shape = obs.shape
        self.is_discrete = False
        self.action_space = np.zeros((2,))

    def prep_obs(self, obs):
        """
        obs: list or ndarray
            the observation returned by the environment
        """
        if not isinstance(obs, list): return obs
        obs = np.asarray(obs[0])
        info = [*obs[1:]]
        return obs, info

    def reset(self):
        obs = self.env.reset()
        obs,_ = self.prep_obs(obs)
        return obs

    def step(self,action):
        """
        action: ndarray (SHAPE = self.action_space.shape)
            the action to take in this step. type can vary depending
            on the environment type
        """
        obs,rew,done,info = self.env.step(action.squeeze())
        obs,targ = self.prep_obs(obs)
        targ[:2] = np.clip(targ[:2],-1,1)
        return obs, rew, done, targ

    def render(self):
        return None

    def close(self):
        self.env.close()

    def make_unity_env(self, env_name, float_params=dict(), time_scale=1,
                                                      seed=time.time(),
                                                      worker_id=None,
                                                      **kwargs):
        """
        creates a gym environment from a unity game

        env_name: str
            the path to the game
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        time_scale: float
            argument to set Unity's time scale. This applies less to
            gym wrapped versions of Unity Environments, I believe..
            but I'm not sure
        seed: int
            the seed for randomness
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        """
        if float_params is None: float_params = dict()
        path = os.path.expanduser(env_name)
        channel = EngineConfigurationChannel()
        env_channel = EnvironmentParametersChannel()
        channel.set_configuration_parameters(time_scale = 1)
        for k,v in float_params.items():
            if k=="validation" and v>=1:
                print("Game in validation mode")
            env_channel.set_float_parameter(k, float(v))
        if worker_id is None: worker_id = seed%500+1
        env_made = False
        n_loops = 0
        worker_id = 0
        while not env_made and n_loops < 50:
            try:
                env = UnityEnvironment(file_name=path,
                                   side_channels=[channel,env_channel],
                                   worker_id=worker_id,
                                   seed=seed)
                env_made = True
            except:
                s = "Error encountered making environment, "
                s += "trying new worker_id"
                print(s)
                worker_id =(worker_id+1+int(np.random.random()*100))%500
                try: env.close()
                except: pass
                n_loops += 1
        env = UnityToGymWrapper(env, allow_multiple_obs=True)
        return env

