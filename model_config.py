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
# preprocess option
config.TRAIN.data_augmentation_ops = ['random_horizontal_flip']
# fine tune checkpoint file
config.TRAIN.fine_tune_checkpoint = None
# train input tf record file
config.TRAIN.train_file_path = "/home/arkenstone/tensorflow/workspace/models/object_detection/models/clothing_detector/train.record"
# eval input tf record file
config.TRAIN.eval_file_path = "/home/arkenstone/tensorflow/workspace/models/object_detection/models/clothing_detector/val.record"
# class annotation file path
config.TRAIN.label_map_file = "model/deepfashion.xml"
# epoch of training input images, default is None (loop infinitely)
config.TRAIN.num_epochs = None
# the num of readers to create
config.TRAIN.num_readers = 4
# if or not to use RandomShuffleQueue to shuffle files and records
config.TRAIN.shuffle = True
# capacity of queue
config.TRAIN.queue_capacity = 600
# Maximum number of elements to store within a queue
config.TRAIN.batch_queue_capacity = 600
# Number of threads to use for batching.
config.TRAIN.num_batch_queue_threads = 8
# Maximum capacity of the queue used to prefetch assembled batches
config.TRAIN.prefetch_queue_capacity = 10
# input training image size
config.TRAIN.im_height = 600
config.TRAIN.im_width = 800
# optimizer during training, only rms_prop_optimizer, momentum_optimizer, adam_optimizer
config.TRAIN.optimizer = "rms_prop_optimizer"
# initial learning rate
config.TRAIN.lr = 0.01
# batch size
config.TRAIN.batch_size = 4
# num of training steps
config.TRAIN.num_steps = 800000
# num of decay steps
config.TRAIN.decay_steps = 100000