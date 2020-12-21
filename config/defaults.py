from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.PROJECT_NAME = 'person-reid-tiny-baseline'  # project name
_C.CFG_NAME = 'baseline'
_C.OUTPUT_DIR = './output/'

# data
_C.DATA = CN()
_C.DATA.DATA_DIR = "/home/xxx/datasets/Market-1501-v15.09.15/"  # dataset path
_C.DATA.INPUT_SIZE = [256, 128]  # HxW
_C.DATA.DATALOADER_NUM_WORKERS = 8  # number of dataloader workers
_C.DATA.SAMPLER = 'triplet'  # batch sampler, option: 'triplet','softmax'
_C.DATA.BATCH_SIZE = 64  # MxN, M: number of persons, N: number of images of per person
_C.DATA.NUM_IMG_PER_ID = 4  # N, number of images of per person

# model
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = "0"  # GPU IDs, i.e. "0,1,2" for multiple GPUs
_C.MODEL.MODEL_NAME = "resnet50"  # backbone name, option: 'resnet50',
_C.MODEL.LAST_STRIDE = 1  # the stride of the last layer of resnet50
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.PRETRAIN_PATH = "/xxx/pretrained_model/resnet50-19c8e357.pth"  # pretrained weight path
_C.MODEL.LOSS_TYPE = 'softmax'  # option: 'triplet+softmax','softmax+center','triplet+softmax+center'
_C.MODEL.LOSS_LABELSMOOTH = 'on'  # using labelsmooth, option: 'on', 'off'
_C.MODEL.COS_LAYER = False


# solver
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = 'Adam'  # optimizer
_C.SOLVER.BASE_LR = 0.00035  # base learning rate

_C.SOLVER.CE_LOSS_WEIGHT = 1.0  # weight of softmax loss
_C.SOLVER.TRIPLET_LOSS_WEIGHT = 1.0  # weight of triplet loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005  # weight of center loss

_C.SOLVER.HARD_FACTOR = 0.0 # harder example mining

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.CENTER_LR = 0.5  # learning rate for the weights of center loss
_C.SOLVER.MARGIN = 0.3  # triplet loss margin

_C.SOLVER.STEPS = [40, 70, 130]
_C.SOLVER.GAMMA = 0.1  # decay factor of learning rate
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCHS = 10  # warm up epochs
_C.SOLVER.WARMUP_METHOD = "linear"  # option: 'linear','constant'


_C.SOLVER.LOG_PERIOD = 50  # iteration of displaying training log
_C.SOLVER.CHECKPOINT_PERIOD = 5  # saving model period
_C.SOLVER.EVAL_PERIOD = 5  # validation period
_C.SOLVER.MAX_EPOCHS = 200  # max training epochs


# test
_C.TEST = CN()
_C.TEST.TEST_IMS_PER_BATCH = 128
_C.TEST.FEAT_NORM = "yes"
_C.TEST.TEST_WEIGHT = './output/resnet50_175.pth'
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.PIDS = "pids.npy"
_C.TEST.CAMIDS = "camids.npy"
_C.TEST.IMG_PATH = "imgpath.npy"
_C.TEST.Q_FEATS = "qfeats.pth"  # query feats
_C.TEST.G_FEATS = "gfeats.pth"  # gallery feats
_C.TEST.TEST_METHOD = 'cosine'
_C.TEST.FLIP_FEATS = 'off'  # using fliped feature for testing, option: 'on', 'off'
_C.TEST.RERANKING = False  # re-ranking
_C.TEST.QUERY_DIR = '/home/xxx/datasets/Market-1501-v15.09.15/query/'
