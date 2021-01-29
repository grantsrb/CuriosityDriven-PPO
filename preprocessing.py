from skimage.color import rgb2grey
import numpy as np

#def pong_prep(pic, env_type='snake-v0'):
#    if "Pong" in env_type:
#        pic = pic[35:195] # crop
#        pic = pic[::2,::2,0] # downsample by factor of 2
#        pic[pic == 144] = 0 # erase background (background type 1)
#        pic[pic == 109] = 0 # erase background (background type 2)
#        pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
#    elif 'Breakout' in env_type:
#        pic = pic[35:195,8:-8] # crop
#        pic = pic[::2,::2,0] # downsample by factor of 2
#        pic = rgb2grey(pic)
#        #pic[pic != 0] = 1
#    elif env_type == "snake-v0":
#        new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
#        new_pic[:,:][pic[:,:,0]==1] = 1
#        new_pic[:,:][pic[:,:,0]==255] = 1.5
#        new_pic[:,:][pic[:,:,1]==255] = 0
#        new_pic[:,:][pic[:,:,2]==255] = .33
#        pic = new_pic
#    return pic[None]

def pong_prep(pic):
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic[None]

def breakout_prep(pic):
    pic = pic[35:195,8:-8] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic = rgb2grey(pic)
    return pic[None]

def snake_prep(pic):
    new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
    new_pic[:,:][pic[:,:,0]==1] = 1
    new_pic[:,:][pic[:,:,0]==255] = 1.5
    new_pic[:,:][pic[:,:,1]==255] = 0
    new_pic[:,:][pic[:,:,2]==255] = .33
    pic = new_pic
    return new_pic[None]

def pendulum_prep(pic):
    return pic.squeeze()[None,:,None,None] # (1,3,1,1)

def center_zero2one(obs):
    """
    obs: ndarray (C, H, W)
        values must range from 0-1
    """
    obs = obs.astype(np.float32).transpose((2,0,1))
    obs = 3*(obs-.5)/.5
    return obs

def grey_centered(obs):
    """
    obs: ndarray (C, H, W)
        values must range from 0-1
    """
    centered = center_zero2one(obs) # (C,H,W)
    obs = centered.mean(0)[None]
    return obs

def null_prep(pic):
    return pic[None]
