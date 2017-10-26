from easydict import EasyDict as edict

config = edict()
# model configuration
config.MODEL = edict()
# input training image size
config.MODEL.IM_HEIGHT = 600
config.MODEL.IM_WIDTH = 800
# nms score threshold
config.MODEL.NMS_THRED = 0.5
# score threshold
config.MODEL.SCORE_THRED = 0.05
# max detections
config.MODEL.MAX_DETECTION = 1000

# training config
config.TRAIN = edict()
# batch size
config.TRAIN.BATCH_SIZE = 4
# initial learning rate
config.TRAIN.LR = 0.01
# momentum
config.TRAIN.MOMENTUM = 0.9
# num of training steps
config.TRAIN.NUM_STEPS = 500000
# preprocess option
config.TRAIN.PREPROCESS_OPS = {'random_horizontal_flip'}
# fine tune checkpoint file
config.TRAIN.FINETUNE_CKPT_PATH = None
# train input tf record file
config.TRAIN.TRAIN_INPUT = "train.record"
# eval input tf record file
config.TRAIN.EVAL_INPUT = "val.record"