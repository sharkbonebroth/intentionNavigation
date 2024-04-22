class EnvParameters:
    OBS_SPACE_SHAPE = (5, 5)
    ACT_SPACE_SHAPE = (2,)

class TrainingParameters:
    lr_actor = 1e-5
    lr_critic = 1e-5
    GAMMA = 0.95 # discount factor
    LAM = 0.95 # for GAE
    EPS_CLIP = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    N_EPOCHS = 10
    N_ENVS = 1
    N_STEPS = 128
    TOTAL_TIMESTEPS = 25000
    ANNEAL_LR = True
    NUM_MINIBATCHES = 48
    MINIBATCH_SIZE = 64
    