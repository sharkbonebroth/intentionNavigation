class NetParameters:
    NET_SIZE = 512
    INTENTION_SIZE = 32
    FOV_SIZE = (35, 35, 2)
    NUM_CHANNEL = 2
    # Intention Length
    VECTOR_LEN = 3  # [intention, action t-1 maybe?] (Primal vector length ) : [dx, dy, d total, action t-1]
    LOAD_MODEL = True
    MODEL_LOAD_PATH = "./models/latest.pth"
    MODEL_SAVE_FOLDER = "./models"
    SAVING_INTERVAL = 200
    
    EVALUATE = True
    NUM_EVALUATION_EPS = 100

class WandbSettings:
    ON = True
    LOGGING_INTERVAL = 30
    EXPERIMENT_NAME = "fancyReward"
    EXPERIMENT_PROJECT = "intentionNav"

class EnvParameters:
    USE_SINGLE_DATA = False
    RENDER = True
    OBS_SPACE_SHAPE = NetParameters.FOV_SIZE
    FOV_SIZE=NetParameters.FOV_SIZE[0]
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
    TOTAL_TIMESTEPS = 50000
    ANNEAL_LR = True
    NUM_MINIBATCHES = 48
    MINIBATCH_SIZE = 64
    