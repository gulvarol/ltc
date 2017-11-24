# Long-term Temporal Convolutions (LTC)

This is the Torch code for the following [paper](https://arxiv.org/abs/1604.04494):

Gül Varol, Ivan Laptev and Cordelia Schmid, Long-term Temporal Convolutions for Action Recognition, PAMI 2017.

Check the [project page](http://www.di.ens.fr/willow/research/ltc/) for more materials.

Contact: [Gül Varol](http://www.di.ens.fr/~varol/).

## Preparation 

1. Install [Torch](https://github.com/torch/distro) with [cuDNN](https://developer.nvidia.com/cudnn) support.

2. Download [UCF101](http://crcv.ucf.edu/data/UCF101.php) and/or [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets.

3. Data pre-processing. TO-DO

4. [C3D model](https://github.com/facebook/C3D) in Torch. TO-DO

## Running the code

You can simply run `th main.lua` to start a training with the default parameters. Following are several examples on how to set parameters in different scenarios:

  ```shell
#### From scratch experiments on UCF101
# Run with default parameters (UCF101 dataset, split 1, 100-frame 58x58 resolution flow network with 0.9 dropout)
th  main.lua -expName flow_100f_d9

# Continue training from epoch 10
th  main.lua -expName flow_100f_d9 -continue -epochNumber 10

# Test final prediction accuracy for model number 20
th  main.lua -expName flow_100f_d9 -evaluate -modelNo 20

# Train 100-frame RGB network from scratch on UCF101 dataset
th  main.lua -nFrames 100 -loadHeight 67  -loadWidth 89  -sampleHeight 58  -sampleWidth 58  \
-stream rgb  -expName rgb_100f_d5  -dataset UCF101 -dropout 0.5 -LRfile LR/UCF101/rgb_d5.lua

# Train 71x71 spatial resolution flow network
th  main.lua -nFrames 100 -loadHeight 81  -loadWidth 108 -sampleHeight 71  -sampleWidth 71  \
-stream flow -expName flow_100f_d5 -dataset UCF101 -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua

# Train 16-frame 112x112 spatial resolution flow network
th  main.lua -nFrames 16  -loadHeight 128 -loadWidth 171 -sampleHeight 112 -sampleWidth 112 \
-stream flow -expName flow_100f_d5 -dataset UCF101 -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua

#### Fine-tune HMDB51 from UCF101
# Train the last layer and freeze the lower layers
th main.lua -expName flow_100f_58_d9/finetune/last             \
-loadHeight 67 -loadWidth 89 -sampleHeight 58 -sampleWidth 58  \
-dataset HMDB51                                                \
-LRfile LR/HMDB51/flow_d9_last.lua                             \
-finetune last                                                 \
-retrain log/UCF101/flow_100f_58_d9/model_50.t7

# Fine-tune the whole network
th main.lua -expName flow_100f_58_d9/finetune/whole            \
-loadHeight 67 -loadWidth 89 -sampleHeight 58 -sampleWidth 58  \
-dataset HMDB51                                                \
-LRfile LR/HMDB51/flow_d9_whole.lua                            \
-finetune whole                                                \
-lastlayer log/HMDB51/flow_100f_58_d9/finetune/last/model_3.t7 \
-retrain log/UCF101/flow_100f_58_d9/model_50.t7

  ```
Note that the results are sensitive to the learning rate (LR) schedule. You can set your own LR by writing a -LRfile. Following are a few observations that can be useful:
- RGB networks converge faster than flow networks.
- High dropout takes longer to converge.
- HMDB51 dataset trains faster.
- Fewer number of frames trains faster.	 


## Pre-trained models
TO-DO

## IDT features
You can find the results of Fisher Vector encoding of the [improved dense trajectory features](http://lear.inrialpes.fr/~wang/improved_trajectories) under the `IDT/` directory.

## Citation
If you use this code, please cite the following:
> @article{varol17a,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TITLE = {{Long-term Temporal Convolutions for Action Recognition}},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUTHOR = {Varol, G{\"u}l and Laptev, Ivan and Schmid, Cordelia},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JOURNAL =  {IEEE Transactions on Pattern Analysis and Machine Intelligence},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YEAR = {2017}  
}

## Acknowledgements
This code is largely built on the ImageNet training example [https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) by [Soumith Chintala](https://github.com/soumith/).

