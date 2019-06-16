## RetinaNet tensorflow version
Unofficial realization of [retinanet](https://arxiv.org/abs/1708.02002) using tf. **NOTE** this project is written for practice, so please don't hesitate to report an issue if you find something run.

TF models [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) have integrated FPN in this framework, and ssd_resnet50_v1_fpn is the synonym of RetinaNet. You could dig into ssd_resnet50_v1_feature_extractor in `models` for coding details. 

Since this work depends on tf in the beginning, I keep only retinanet backbone, loss and customed retinanet_feature_extractor in standard format. To make it work, here are the steps: 
- download tensorflow [models](https://github.com/tensorflow/models) and install object detection api following [this way](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). 
- replace `model_builder.py` under `builder` with this one. 
- put `retinanet_feature_extractor.py` under `models`
- put `retinanet.py` under `object detection` root path
- modify `retinanet_50_train.config` and `train.sh` with your customed settings and data inputs. Then run `train.sh` to start training.