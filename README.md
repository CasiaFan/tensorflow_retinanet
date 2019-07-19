## RetinaNet tensorflow version
Unofficial realization of [retinanet](https://arxiv.org/abs/1708.02002) using tf. **NOTE** this project is written for practice, so please don't hesitate to report an issue if you find something run.

TF models [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) have integrated FPN in this framework, and ssd_resnet50_v1_fpn is the synonym of RetinaNet. You could dig into ssd_resnet50_v1_feature_extractor in `models` for coding details. 

Since this work depends on tf in the beginning, I keep only retinanet backbone, loss and customed retinanet_feature_extractor in standard format. To make it work, here are the steps: 
- Download tensorflow [models](https://github.com/tensorflow/models) and install object detection api following [this way](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). 
- Add retinanet feature extractor to `model_builder.py`: 
```python3
from object_detection.models.retinanet_feature_extractor import RetinaNet50FeatureExtractor, RetinaNet101FeatureExtractor

SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    ...
    'retinanet_50': RetinaNet50FeatureExtractor,
    'retinanet_101': RetinaNet101FeatureExtractor,
}
```
- Put `retinanet_feature_extractor.py` and `retinanet.py` under `models`
- Modify `retinanet_50_train.config` and `train.sh` with your customed settings and data inputs. Then run `train.sh` to start training.