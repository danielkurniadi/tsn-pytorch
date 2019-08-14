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
python3 <SOURCE_VIDEO_DIR> <OUTPUT_VIDEO_DIR> -j 8 --buffer_size 24 --img_ext .jpg

```

*  SOURCE_VIDEO_DIR: your video dataset folder to be preprocessed 
*  OUTPUT_VIDEO_DIR: your output folder for preprocessed result
*  -j or --n_jobs: number of processes
* -b or --buffer_size: step size of video frames, default=24
* --img_ext: img extension format for output frames; choice: [.jpg/.png/.jpeg]

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

*  OUTPUT_VIDEO_DIR: your output folder for preprocessed result
* FOLDER_FOR_SPLIT_FILES: folder to write split files
*  -k or --n_splits: number of splitting
* --split_prefix: naming prefix to prepend on split files


Note that `<OUTPUT_VIDEO_DIR>` can be any dataset folder with the same structure you prefer to train on.


## Training

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py hmdb51 Flow <hmdb51_rgb_train_list> <hmdb51_rgb_val_list>  \
   --arch BNInception --num_segments 3 --ext .jpg -b 48 --img_prefix rgb --lr 0.001 --lr_steps 20 40 \
   --gd 20 --epochs 80 --eval-freq 1 --print-freq 5 --snapshot_pref hmdb51_bninception_rgb
```

For flow models:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py hmdb51 Flow <hmdb51_flow_train_list> <hmdb51_flow_val_list>  \
   --arch BNInception --num_segments 3 --ext .jpg -b 48 --flow_prefix flow --lr 0.001 --lr_steps 20 40 \
   --gd 20 --epochs 80 --eval-freq 1 --print-freq 5 --snapshot_pref hmdb51_bninception_flow
```

For RGB-diff models:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py hmdb51 RGBDiff <hmdb51_rgbdiff_train_list> <hmdb51_rgbdiff_val_list>  \
   --arch BNInception --num_segments 3 --ext .jpg -b 48 --img_prefix rgb --lr 0.001 --lr_steps 20 40 \
   --gd 20 --epochs 80 --eval-freq 1 --print-freq 5 --snapshot_pref hmdb51_bninception_ 
```

For ARP-diff

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 test.py <DATASET_NAME> RGB <SPLIT_FILES_FOR_TEST> \
   <CHECKPOINT_PTH_FILE> --arch BNInception --img_prefix <IMG_PREFIX_OF_DATASET_FRAMES> --ext <.png|.jpg|.jpeg> \
   -b <BATCH_SIZE> --print_freq 1 --num_segments <NUM_SEGMENTS_PER_VIDEO>

```

Or for flow models:
 
```bash
CUDA_VISIBLE_DEVICES=1,2 python3 test.py <DATASET_NAME> Flow <SPLIT_FILES_FOR_TEST> \
   <CHECKPOINT_PTH_FILE> --arch BNInception --flow_prefix <FLOW_PREFIX_OF_DATASET_FRAMES> --ext <.png|.jpg|.jpeg> \
   -b <BATCH_SIZE> --print_freq 1 --num_segments <NUM_SEGMENTS_PER_VIDEO>
```


## Adding Custom Dataset:
When adding custom dataset, you can name your dataset with certain namings, say "mydataset". 
Then what you need to do:

1. Go to `opts.py` files and look for argument parser: `modality` (line 4). Add you own dataset naming to the `choices` kwargs
2. To train on custom dataset, look for `main()` function in `main.py`. Add if-else statement and speficy number of classes in custom dataset
3. To test on custom dataset, look for `test()` function in `test.py`. Add if-esle statement and specify number of classes in custom dataset

