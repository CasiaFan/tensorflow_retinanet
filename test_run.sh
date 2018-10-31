#!/usr/bin/env bash
python3 model_main.py --model_dir="model1" \
                      --train_file_pattern="data/items_train.record" \
                      --eval_file_pattern="data/items_val.record" \
                      --image_size=224 \
                      --num_classes=1 \
                      --label_map_path="data/label.pbtxt"