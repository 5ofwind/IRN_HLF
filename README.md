
Video Super-Resolution with Inverse Recurrent Net and Hybrid Local Fusion,
Dingyi Li, Zengfu Wang and Jian Yang,
Neurocomputing, 2022

The code is based on the code of EDVR, RRN and RSDN.

For testing, use:
python IRN_2stages.py (test IRN_2stages)
python IRN_3stages.py (test IRN_3stages)
python EDVR/BD_test_Vid4_REDS4_with_GT.py (test EDVR_BD)
matlab: fuse.m (for IRN_3stages_parallel)
python EDVR/BD_test_Vid4_REDS4_with_GT_2_Stage2.py (test IRN_3stages_parallel_cascade)
python RSDN.py (test RSDN_BD)

For training, use:
python train.py (train IRN_3stages)
python train_cascade.py (train IRN_3stages_cascade and also use it for IRN_3stages_parallel_cascade)
