# TSN-Pytorch

*Now in experimental release, suggestions welcome*.

**Note**: always use `git clone --recursive https://github.com/yjxiong/tsn-pytorch` to clone this project. 
Otherwise you will not be able to use the inception series CNN archs. 

This is a reimplementation of temporal segment networks (TSN) in PyTorch. All settings are kept identical to the original caffe implementation.

For optical flow extraction and video list generation, you still need to use the original [TSN codebase](https://github.com/yjxiong/temporal-segment-networks).


## Dependencies

* Python3 (3.5>=)
* OpenCV Python (OpenCV contrib 3.4>=)
* Pytorch (1.0>=)

### Pip installation
```
pip3 install -r requirements.pip
```


## Preprocessing

### Approximated Rank Pooling
To convert video dataset to approximated rank pooled frames, use the `convert_to_ARP.py` scripts.

The command to reproduce the Rank pooling preprocessing is as follows:

```bash
python3 <SOURCE_VIDEO_DIR> <OUTPUT_VIDEO_DIR> -j 8 --img_ext .jpg

```

Note that it assumes your video dataset folder is structured as follows:
```
|- dataset_dir/
   |- class_folder_A
      |- video0001.avi
      |- video0002.avi

      ...
   |- class_folder_B
      |- video0004.avi
      |- video0005.avi
      ...
```

The output will be populated by preprocessed frames. The folder structure of output directory is as follows:
```
|- dataset_dir/
   |- class_folder_A
         |- folder_of_video0001
         |- folder_of_video0002
            |- frame0001.png
            |- frame0002.png
            |- frame0003.png
         ...
   |- class_folder_B
         |- folder_of_video0004
         |- folder_of_video0005
         ...
```


The above folder structure is common in data science.

## Dense Flow

To convert video dataset to approximated rank pooled frames, use the `denseFlow_gpu` executables. 
You can download and install it from here: https://github.com/wanglimin/dense_flow


## Spliting Dataset
Assuming preprocessing has been done to your `OUTPUT_VIDEO_DIR`, you can run cross validation split, using `split_data.py` script.

```
python3 <OUTPUT_VIDEO_DIR> <FOLDER_FOR_SPLIT_FILES> --n_splits 5 --split_prefix mydata_split
```

Note that `<OUTPUT_VIDEO_DIR>` can be any dataset folder with the same structure you prefer to train on.


## Training

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For flow models:

```bash
python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_  
```

For RGB-diff models:

```bash
python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For ARP-diff

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB <ucf101_rgb_val_list> ucf101_bninception_rgb_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name>

```

Or for flow models:
 
```bash
python test_models.py ucf101 Flow <ucf101_rgb_val_list> ucf101_bninception_flow_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name> --flow_pref flow_

```
