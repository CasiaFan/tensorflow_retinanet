from easydict import EasyDict as edict

config = edict()
# model configuration
config.MODEL = edict()
# nms score threshold
config.MODEL.nms_thred = 0.5
# score threshold
config.MODEL.score_thred = 0.05
# max detections
config.MODEL.max_detection = 1000

# training config
config.TRAIN = edict()
# batch size
config.TRAIN.batch_size = 4
# initial learning rate
config.TRAIN.lr = 0.01
# momentum
config.TRAIN.momentum = 0.9
# num of training steps
config.TRAIN.num_steps = 500000
# preprocess option
config.TRAIN.data_augmentation_ops = ['random_horizontal_flip']
# fine tune checkpoint file
config.TRAIN.fine_tune_ckpt_path = ""
# train input tf record file
config.TRAIN.train_file_path = "train.record"
# eval input tf record file
config.TRAIN.eval_file_path = "val.record"
# class annotation file path
config.TRAIN.label_map_file = "model/deepfashion.xml"
# epoch of training input images, default is None (loop infinitely)
config.TRAIN.num_epochs = None
# the num of readers to create
config.TRAIN.num_readers = 4
# if or not to use RandomShuffleQueue to shuffle files and records
config.TRAIN.shuffle = True
# capacity of queue
config.TRAIN.capacity = 600
# Maximum number of elements to store within a queue
config.TRAIN.batch_queue_capacity = 600
# Number of threads to use for batching.
config.TRAIN.num_batch_queue_threads = 8
# Maximum capacity of the queue used to prefetch assembled batches
config.TRAIN.prefetch_queue_capacity = 10
# input training image size
config.TRAIN.im_height = 600
config.TRAIN.im_width = 800