from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set 
import pdb
from torch.optim import lr_scheduler 
import socket
import time
import cv2
import math
import sys
from utils import Logger 
import numpy as np
from arch import RSDN9_128

from EDVR.models.modules.EDVR_arch import EDVR
from collections import OrderedDict

import datetime
import torchvision.utils as vutils
import random
from loss import CharbonnierLoss

import itertools

parser = argparse.ArgumentParser(description='PyTorch RSDN')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')#70#10
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots. This is a savepoint, using to save training model.')#10
parser.add_argument('--lr', type=float, default=0.000005, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt', help='where record all of image name in dataset.')
parser.add_argument('--patch_size', type=int, default=0, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--layer', type=int, default=10, help='network layer')#5
parser.add_argument('--stepsize', type=int, default=1, help='Learning rate is decayed by a factor of 10 every half of total epochs')#60#1
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')#0.1
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='Location to save checkpoint models')
parser.add_argument('--save_train_log', type=str ,default='./result/log/')
parser.add_argument('--weight-decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--log_name', type=str, default='rsdn')
parser.add_argument('--other_dataset', type=bool, default=False, help="If True using vid4k(test),else using vimo90k")
parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 

parser.add_argument('--pretrain', type=str, default='../IRN.pth')
parser.add_argument('--pretrain_EDVR', type=str, default='../EDVR/experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth')

from data import get_test_set 
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--test_dir',type=str,default='../datasets/udm10')
parser.add_argument('--save_test_log', type=str,default='./log/test')
parser.add_argument('--image_out', type=str, default='./out/')
#parser.add_argument('--gpu-devices2', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 
import warnings
warnings.filterwarnings("ignore")

opt = parser.parse_args()
opt.data_dir = '../datasets/vimeo90k/vimeo_septuplet/sequences'
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

#def load(self):
#    load_path_G = self.opt['path']['pretrain_model_G']
#    if load_path_G is not None:
#        logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
#        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])


def save_network(network, network_label, iter_label):
    save_filename = '{}.pth'.format(iter_label)
    save_path = './result/weight/rsdn/EDVR_X4_10L_0_epoch_' + save_filename
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

#def save(self, iter_label):
#    self.save_network(self.edvr, 'G', iter_label)


def main():
    #torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_'+opt.log_name+'.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
       use_gpu = torch.cuda.is_available()

    #if use_gpu:
    #    cudnn.benchmark = True
        #torch.cuda.manual_seed_all(opt.seed)

    seed=opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

    pin_memory = True if use_gpu else False

    print(opt)
    print('===> Loading Datasets')
    train_set = get_training_set(opt.data_dir, opt.scale, opt.data_augmentation, opt.file_list) 
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True, pin_memory=pin_memory, drop_last=True)
    print('===> DataLoading Finished')
    # Selecting network layer
    #n_c = 128
    n_d = 16
    n_b = 10
    rsdn = RSDN9_128(4) # initial filter generate network 

    edvr = EDVR(128, 5, 8, 5, 20, HR_in=True)

    p = sum(p.numel() for p in rsdn.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))
    print(rsdn)
    print('===> {}L model has been initialized'.format(n_b))
    rsdn = torch.nn.DataParallel(rsdn)
    edvr = torch.nn.DataParallel(edvr)
    criterion = nn.L1Loss(reduction='sum')


    if os.path.isfile(opt.pretrain):
        rsdn.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage),False)
        load_network(opt.pretrain_EDVR, edvr, False)#True for parallel local fusion
        #edvr.load_state_dict(torch.load(opt.pretrain_EDVR),strict=True)

#, map_location=lambda storage, loc: storage
        print('===> pretrained model is load')
    else:
        raise Exception('pretrain model is not exists')

    if use_gpu:
        rsdn = rsdn.cuda()
        edvr = edvr.cuda()
        criterion = criterion.cuda()

    '''
    normal_params = []
    tsa_fusion_params = []
    for k, v in rsdn.named_parameters():
        if v.requires_grad:
            if 'neuro.SD9' in k:

                tsa_fusion_params.append(v)
            else:
                normal_params.append(v)
        #else:
        #    if self.rank <= 0:
        #        logger.warning('Params [{:s}] will not optimize.'.format(k))
    optim_params = [
        {  # add normal params first
            'params': normal_params,
            #'lr': train_opt['lr_G']
        },
        {
            'params': tsa_fusion_params,
            #'lr': train_opt['lr_G']
        },
    ]
    
    optimizer = optim.Adam(optim_params, lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay) 
#rsdn.parameters()
    '''
#,edvr.parameters()
    optimizer = optim.Adam(edvr.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay) 
#rsdn.parameters(),rsdn.parameters(),
#itertools.chain()
    if opt.stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=opt.gamma)
        #.CosineAnnealingLR(optimizer, 1, eta_min=0, last_epoch=-1)#.StepLR(optimizer, step_size = opt.stepsize, gamma=opt.gamma)
    #optimizer.param_groups[0]['lr'] = 0#normal
    #optimizer.param_groups[1]['lr'] = 0.00001#TSA special


    alter=0

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        ttttt=opt.test_dir



        alter=train(train_loader, rsdn, edvr, opt.scale, criterion, optimizer, epoch, use_gpu, ttttt, alter) #fed data into network



        scheduler.step()
        if (epoch) % (opt.snapshots) == 0:
            checkpoint(rsdn, epoch)
            save_network(edvr, 'G', epoch)
            #checkpoint_EDVR(edvr, epoch)





def train(train_loader, rsdn, edvr, scale, criterion, optimizer, epoch, use_gpu, ttttt, alter):
    train_mode = True
    epoch_loss = 0
    #rsdn.train()
    edvr.train()

    average_loss=0
    total_time=0

    print(time.asctime(time.localtime(time.time())))
    print('---------------------------------------------------------------------')
    for iteration, data in enumerate(train_loader):

        #print(iteration)
        num_test=5000
        #if (((iteration) % num_test == 0) and ((epoch==1 and iteration==0)==False)):
        if ((iteration) % num_test == 0):            
            checkpoint(rsdn, epoch)
            save_network(edvr, 'G', epoch)#.save(iteration)#self.save_network(self.edvr, 'G', iter_label)

            #checkpoint_EDVR(edvr, epoch)
            
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), average_loss/num_test, total_time))
            print(time.asctime(time.localtime(time.time())))

            average_loss=0
            total_time=0

            '''
            print('===> Vid4 testing dataset:')
            PSNR_avg = 0
            SSIM_avg = 0
            count = 0
            out = []
            #scene_list = ['archpeople'] # UDM10
            scene_list = ['calendar','city','foliage','walk'] # Vid4
            #scene_list = ['city'] # Vid4
            #scene_list = ['archpeople','archwall','auditorium','band','caffe','camera','lake','clap','photography','polyflow'] # UDM10
            #scene_list = ['car05_001', 'hdclub_003_001', 'hitachi_isee5_001', 'hk004_001', 'HKVTG_004', 'jvc_009_001', 'NYVTG_006', 'PRVTG_012', 'RMVTG_011', 'veni3_011', 'veni5_015'] # SPMCS

            for scene_name in scene_list:
                test_set = get_test_set('../datasets/Vid4', opt.scale, scene_name)
                test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, pin_memory=True, drop_last=False)#pin_memory=pin_memory
                #print('===> DataLoading Finished')
                PSNR, SSIM, out = test(test_loader, rsdn, edvr, opt.scale, scene_name)
                PSNR_avg += PSNR
                SSIM_avg += SSIM
                count += 1
            PSNR_avg = PSNR_avg/len(scene_list)
            SSIM_avg = SSIM_avg/len(scene_list)
            print('======================================> Average PSNR = {:.6f}'.format(PSNR_avg))
            print('======================================> Average SSIM = {:.6f}'.format(SSIM_avg))


            print(time.asctime(time.localtime(time.time())))
            

            
            print('===> UDM10 testing dataset:')
            PSNR_avg = 0
            SSIM_avg = 0
            count = 0
            out = []
            #scene_list = ['archpeople'] # UDM10
            #scene_list = ['calendar','city','foliage','walk'] # Vid4
            #scene_list = ['city'] # Vid4
            scene_list = ['archpeople','archwall','auditorium','band','caffe','camera','lake','clap','photography','polyflow'] # UDM10
            #scene_list = ['archpeople']
            #scene_list = ['car05_001', 'hdclub_003_001', 'hitachi_isee5_001', 'hk004_001', 'HKVTG_004', 'jvc_009_001', 'NYVTG_006', 'PRVTG_012', 'RMVTG_011', 'veni3_011', 'veni5_015'] # SPMCS

            for scene_name in scene_list:
                test_set = get_test_set(ttttt, opt.scale, scene_name)
                test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, pin_memory=True, drop_last=False)#pin_memory=pin_memory
                #print('===> DataLoading Finished')
                PSNR, SSIM, out = test(test_loader, rsdn, edvr, opt.scale, scene_name)
                PSNR_avg += PSNR
                SSIM_avg += SSIM
                count += 1
            PSNR_avg = PSNR_avg/len(scene_list)
            SSIM_avg = SSIM_avg/len(scene_list)
            print('======================================> Average PSNR = {:.6f}'.format(PSNR_avg))
            print('======================================> Average SSIM = {:.6f}'.format(SSIM_avg))


            print(time.asctime(time.localtime(time.time())))
            '''




        #x_input, target = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        #prediction_final=(prediction+prediction2)/2
        LR, LR_d, LR_s, target, L, target2, target3 = data[0],data[1], data[2], data[3], data[4], data[5], data[6]
        optimizer.zero_grad()
        #optimizer2.zero_grad()
        B, _, T, _ ,_ = LR.shape
        out = []
        out2 = []
        out3 = []
        if 1==1:
            LR = Variable(LR).cuda()
            LR_d = Variable(LR_d).cuda()
            LR_s = Variable(LR_s).cuda()
            target = Variable(target).cuda()
            target2 = Variable(target2).cuda()
            target3 = Variable(target3).cuda()

            #print(LR.shape)

            #print(prediction_EDVR.shape)


            t0 = time.time()

            #prediction_EDVR=edvr(LR)

            LR2=flip(LR,1)
            LR_d2=flip(LR_d,1)
            LR_s2=flip(LR_s,1)
            
            LR = torch.cat((LR, LR2, LR), dim=1)
            LR_d = torch.cat((LR_d, LR_d2, LR_d), dim=1)
            LR_s = torch.cat((LR_s, LR_s2, LR_s), dim=1)
            





            prediction, out_d, out_s = rsdn(LR, LR_d, LR_s)



          
            #print(prediction.shape[1]/3)

            shape_prediction=int(prediction.shape[1]/3)
            #prediction=(prediction[:,0:shape_prediction,:,:,:]+flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1)+prediction[:,shape_prediction*2:shape_prediction*3,:,:,:])/3
            
            prediction_final=(flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1)+prediction[:,shape_prediction*2:shape_prediction*3,:,:,:])/2
            #print(prediction_final.shape)

            #prediction_final=(prediction_final[:,3,:,:,:]+prediction_EDVR)/2
            prediction_final=edvr(prediction_final[:,1:-1,:,:,:])#[:,1:-1,:,:,:]





        '''
        #loss1=criterion(flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1), target)/(B*T)
        #loss2=criterion(prediction[:,shape_prediction*2:shape_prediction*3,:,:,:], target)/(B*T)
        #loss3=criterion(prediction_final, target)/(B*T)

        loss1=1/3*(criterion(flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1), target)/(B*T)
+criterion(flip(out_d[:,shape_prediction:shape_prediction*2,:,:,:],1), target2)/(B*T)
+criterion(flip(out_s[:,shape_prediction:shape_prediction*2,:,:,:],1), target3)/(B*T))

        loss2=1/3*(criterion(prediction[:,shape_prediction*2:shape_prediction*3,:,:,:], target)/(B*T)
+criterion(out_d[:,shape_prediction*2:shape_prediction*3,:,:,:], target2)/(B*T)
+criterion(out_s[:,shape_prediction*2:shape_prediction*3,:,:,:], target3)/(B*T))

        #loss3=1/3*(criterion(prediction[:,shape_prediction*2:shape_prediction*3,:,:,:], target)/(B*T)
#+criterion(out_d[:,shape_prediction*2:shape_prediction*3,:,:,:], target2)/(B*T)
#+criterion(out_s[:,shape_prediction*2:shape_prediction*3,:,:,:], target3)/(B*T))
        
        if alter==0:
            alter=1
            loss=loss1
        elif alter==1:
            alter=2
            loss=loss2
        else:
            out_d=(flip(out_d[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_d[:,shape_prediction*2:shape_prediction*3,:,:,:])/2
            out_s=(flip(out_s[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_s[:,shape_prediction*2:shape_prediction*3,:,:,:])/2
            loss3=1/3*(criterion(prediction_final, target)/(B*T)+criterion(out_d, target2)/(B*T)+criterion(out_s, target3)/(B*T))
#+0.1*loss4

            alter=0
            loss=loss3
        '''
        
        #loss4=criterion(flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1),
#prediction[:,shape_prediction*2:shape_prediction*3,:,:,:])/(B*T)
        #loss3=criterion(prediction_final, target)/(B*T)+criterion(out_d, target2)/(B*T)
        
        #loss=loss3#+0.1*loss4#(loss1+loss2)/2#(loss1+loss2)/2#(loss1+loss2+loss3)/3#loss3#(0.2*loss1+0.2*loss2+0.6*loss3)######loss3
        #loss=criterion(prediction_final, target)/(B*T)
        
        out_d=(flip(out_d[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_d[:,shape_prediction*2:shape_prediction*3,:,:,:])/2
        out_s=(flip(out_s[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_s[:,shape_prediction*2:shape_prediction*3,:,:,:])/2


        loss=criterion(prediction_final, target[:,3,:,:,:])/(B)#+1/3*(criterion(prediction_final, target)/(B*T)+criterion(out_d, target2)/(B*T)+criterion(out_s, target3)/(B*T))

        #loss=1/3*(criterion(prediction_final, target[:,3,:,:,:])/(B*T)+criterion(out_d[:,3,:,:,:], target2[:,3,:,:,:])/(B*T)+criterion(out_s[:,3,:,:,:], target3[:,3,:,:,:])/(B*T))#1/3*
#+0.1*loss4



        #loss=criterion(prediction_final, target)/(B*T)
        loss.backward()

        #print(torch.norm(rsdn.parameters(),2))
        #torch.nn.utils.clip_grad_norm(edvr.parameters(),1)
        #optimizer2.step()
        optimizer.step()
        epoch_loss += loss.item()
        average_loss+=loss.item()
        t1 = time.time()
        total_time+=t1-t0
        
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))

    return alter






def checkpoint(rsdn, epoch): 
    save_model_path = os.path.join(opt.save_model_path,'rsdn')#, systime
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rsdn.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))





def checkpoint_EDVR(rsdn, epoch): 
    save_model_path = os.path.join(opt.save_model_path,'rsdn')#, systime
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'EDVR_X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rsdn.state_dict(), os.path.join(save_model_path, model_name))
    
    print('EDVR_Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True







def test(test_loader, rsdn, edvr, scale, test_name):
    train_mode = False
    rsdn.eval()
    edvr.eval()
    count = 0
    PSNR = 0
    SSIM = 0
    PSNR_ = 0
    SSIM_ = 0
    out = []
    out2 = []
    out3 = []
    out4 = []
    out5 = []
    for image_num, data in enumerate(test_loader):
        LR, LR_d, LR_s, target, L = data[0],data[1], data[2], data[3], data[4]#, data[5], data[6]
        with torch.no_grad():
            LR = Variable(LR).cuda()
            LR_d = Variable(LR_d).cuda()
            LR_s = Variable(LR_s).cuda()
            target = Variable(target).cuda()
            #target2 = Variable(target2).cuda()
            #target3 = Variable(target3).cuda()

            t0 = time.time()


            #prediction_EDVR=edvr(LR)


            LR2=flip(LR,1)
            LR_d2=flip(LR_d,1)
            LR_s2=flip(LR_s,1)
            
            LR = torch.cat((LR, LR2, LR), dim=1)
            LR_d = torch.cat((LR_d, LR_d2, LR_d), dim=1)
            LR_s = torch.cat((LR_s, LR_s2, LR_s), dim=1)
            
            #print(LR.shape)




            prediction, out_d, out_s = rsdn(LR, LR_d, LR_s)
          
            #print(prediction.shape[1]/3)

            shape_prediction=int(prediction.shape[1]/3)
            #prediction=(prediction[:,0:shape_prediction,:,:,:]+flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1)+prediction[:,shape_prediction*2:shape_prediction*3,:,:,:])/3
            
            prediction=(flip(prediction[:,shape_prediction:shape_prediction*2,:,:,:],1)+prediction[:,shape_prediction*2:shape_prediction*3,:,:,:])/2

            #prediction=(prediction+prediction_EDVR)/2
            #prediction=edvr(prediction)

            #out_d=(flip(out_d[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_d[:,shape_prediction*2:shape_prediction*3,:,:,:])/2

            #out_s=(flip(out_s[:,shape_prediction:shape_prediction*2,:,:,:],1)+out_s[:,shape_prediction*2:shape_prediction*3,:,:,:])/2

            
            #print(out_d.shape)

            #print(out_s.shape)

            '''
            LR=flip(LR,1)
            LR_d=flip(LR_d,1)
            LR_s=flip(LR_s,1)

            prediction2, out_d, out_s = filter_net(LR, LR_d, LR_s)
            prediction2=flip(prediction2,1)

            #print(prediction2.shape)
            #print(LR.shape)
            prediction=(prediction+prediction2)/2
            '''
            t1 = time.time()
            #print("===> Timer: %.4f sec." % (t1 - t0))  

        count += 1
        prediction = prediction.squeeze(0).permute(0,2,3,1) # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr 

        L = L.numpy()
        L = int(L)
        target = target.squeeze(0).permute(0,2,3,1) # [T,H,W,C]
        target = target.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr

        #for i in range(L):
        #    #save_img(prediction[i], test_name, i)
        target = crop_border_RGB(target, 8)
        prediction = crop_border_RGB(prediction, 8)
        for i in range(L):
            # test_Y______________________
            prediction_Y = bgr2ycbcr(prediction[i])
            target_Y = bgr2ycbcr(target[i])
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255
            # _______________________________
            #prediction_Y = prediction[i] * 255
            #target_Y = target[i] * 255
            # ________________________________
            # calculate PSNR and SSIM
            #print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(calculate_psnr(prediction_Y, target_Y), calculate_ssim(prediction_Y, target_Y)))
            PSNR += calculate_psnr(prediction_Y, target_Y)
            SSIM += calculate_ssim(prediction_Y, target_Y)
            out.append(calculate_psnr(prediction_Y, target_Y))
        #print('===>{} PSNR = {}'.format(test_name, PSNR/(L)))
        #print('===>{} SSIM = {}'.format(test_name, SSIM/(L)))
        PSNR_ += PSNR/(L)
        SSIM_ += SSIM/(L)
    return PSNR_, SSIM_, out




def save_img(prediction, scene_name, image_num):
    save_dir = os.path.join(opt.image_out, 'aaa')#, systime
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_dir = os.path.join(save_dir, '{}_{:03}'.format(scene_name, image_num+1) + '.png')
    #cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def crop_border_Y(prediction, shave_border=0):
    prediction = prediction[shave_border:-shave_border, shave_border:-shave_border]
    return prediction

def crop_border_RGB(target, shave_border=0):
    target = target[:,shave_border:-shave_border, shave_border:-shave_border,:]
    return target

def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')





if __name__ == '__main__':
    main()    
