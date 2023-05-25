# IRN-HLF

This is the code for our paper:

Li, Dingyi, Zengfu Wang, and Jian Yang. "Video super-resolution with inverse recurrent net and hybrid local fusion." Neurocomputing 489 (2022): 40-51.

https://www.sciencedirect.com/science/article/pii/S0925231222002880

The code is based on PyTorch, and the code of EDVR, RRN and RSDN.

For testing, use:

python test_IRN_2stages.py (test IRN_2stages)

python test_IRN_3stages.py (test IRN_3stages)

python EDVR/BD_test_Vid4_REDS4_with_GT.py (test EDVR_BD)

matlab: fuse.m (for IRN_3stages_parallel)

python EDVR/BD_test_Vid4_REDS4_with_GT_2_Stage2.py (test IRN_3stages_parallel_cascade)

python test_RSDN.py (test RSDN_BD)

For training, use:

python train.py (train IRN_3stages)

python train_cascade.py (train IRN_3stages_cascade and also use it for IRN_3stages_parallel_cascade)

The models for our IRN, EDVR and RSDN can be found at 

https://pan.baidu.com/s/1sinco1rBm3ptdYAV9TH1ow

The password is: iraj
