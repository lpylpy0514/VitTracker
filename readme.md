# VitTracker for OpenCV 4.9.0

opencv: https://github.com/opencv/opencv/blob/4.x/modules/video/src/tracking/tracker_vit.cpp

opencv_zoo: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack

# About training

## Install the environment

Use the Anaconda (CUDA 11.1)

```
conda create -n py38 python=3.8
conda activate py38
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
sh install.sh
```


## Set project paths

Run the following command to set paths for this project

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files

```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation

Put the tracking datasets in ./data. It should look like:

   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training

```
python tracking/train.py --script vit_dist --config vit_48_h32_noKD --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb 0
```

## Evaluation

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/vit_dist`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:

- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)

```
python tracking/test.py vit_dist vit_48_h32_noKD --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```

- GOT10K-test

```
python tracking/test.py vit_dist vit_48_h32_noKD --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name vit_dist --cfg_name vit_48_h32_noKD
```

- TrackingNet

```
python tracking/test.py vit_dist vit_48_h32_noKD --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name vit_dist --cfg_name vit_48_h32_noKD
```

## Export to ONNX

```sh
python tracking/onnxexport --script vit_dist --config vit_48_h32_noKD
python tracking/onnxsimplify.py
```

## Contact me

Pengyu Liu  liupengyu@mail.dlut.edu.cn