# Video Super-Resolution with Inverse Recurrent Net and Hybrid Local Fusion (IRN-HLF)

This is the code for our paper:

Li, Dingyi, Zengfu Wang, and Jian Yang. "Video super-resolution with inverse recurrent net and hybrid local fusion." Neurocomputing 489 (2022): 40-51.

https://www.sciencedirect.com/science/article/pii/S0925231222002880

The code is based on PyTorch, and the code of EDVR, RRN and RSDN.

# Abstract

Video super-resolution converts low-resolution videos to sharp high-resolution ones. In order to make better use of temporal information in video super-resolution, we design inverse recurrent net and hybrid local fusion. We concatenate the original low-resolution input sequence and its inverse sequence repeatedly. The new sequence is viewed as a combination of different stages, and is processed sequentially by using our inverse recurrent net. The outputs of the last two stages in opposite directions are fused to generate the final images. Our inverse recurrent net can extract more bidirectional temporal information in the input sequence, without adding parameter to the corresponding unidirectional recurrent net. We also propose a hybrid local fusion method which uses parallel fusion and cascade fusion for incorporating sliding-window-based methods into our inverse recurrent net. Extensive experimental results demonstrate the effectiveness of the proposed inverse recurrent net and hybrid local fusion, in terms of visual quality and quantitative evaluations.

# For testing, use:

python test_IRN_2stages.py #(test IRN_2stages)

python test_IRN_3stages.py #(test IRN_3stages)

python EDVR/BD_test_Vid4_REDS4_with_GT.py #(test EDVR_BD)

matlab: fuse.m #(for IRN_3stages_parallel)

python EDVR/BD_test_Vid4_REDS4_with_GT_2_Stage2.py #(test IRN_3stages_parallel_cascade)

python test_RSDN.py #(test RSDN_BD for comparison)

# For training, use:

python train.py #(train IRN_3stages)

python train_cascade.py #(train IRN_3stages_cascade and also use it for IRN_3stages_parallel_cascade)

# The models for our IRN, EDVR and RSDN can be found at 

https://pan.baidu.com/s/1sinco1rBm3ptdYAV9TH1ow

The password is: iraj
