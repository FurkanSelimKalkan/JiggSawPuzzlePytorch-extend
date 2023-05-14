# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""

import torch
import torchaudio as torchaudio
import torchvision

from JigsawPuzzlePytorch.Utils.logger import Logger

print(torch.cuda.is_available())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print(f"Using device: {device}")

import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')
from Utils.logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('Dataset')
from JigsawNetwork import Network

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy


parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
#parser.add_argument('data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
#parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--batch', default=92, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=6, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

#from ImageDataLoader import DataLoader
from Dataset.JigsawImageLoader import DataLoader


def main():
    if args.gpu is not None:
        print(('Using GPU %d'%args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')

    print('Process number: %d'%(os.getpid()))

    ## DataLoader initialize ILSVRC2012_train_processed
    #trainpath = 'ILSVRC2012_img_train_t3'
    trainpath = 'C:/Users/furka/Desktop/ki/ILSVRC2012_img_train_t3'
    if os.path.exists(trainpath+'_255x255'):
        trainpath += '_255x255'
    train_data = DataLoader(trainpath,'C:/Users/furka/Desktop/ki/ilsvrc12_train_70.txt',
                            classes=args.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            num_workers=args.cores)

    #valpath = '/ILSVRC2012_img_val'
    valpath = 'C:/Users/furka/Desktop/ki/ILSVRC2012_img_train_t3'
    if os.path.exists(valpath+'_255x255'):
        valpath += '_255x255'
    val_data = DataLoader(valpath, 'C:/Users/furka/Desktop/ki/ilsvrc12_val_15.txt',
                            classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            num_workers=args.cores)
    N = train_data.N

    iter_per_epoch = train_data.N/args.batch
    print('Images: train %d, validation %d'%(train_data.N,val_data.N))

    # Network initialize
    net = Network(args.classes)
    if args.gpu is not None:
        net.cuda()

    ############## Load from checkpoint if exists, otherwise from model ###############
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files)>0:
            files.sort()
            #print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ',ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)

    logger = Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint+'/test')

    ############## TESTING ###############
    if args.evaluate:
        test(net,criterion,None,val_loader,0)
        return

    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,args.classes)))
    print(('Checkpoint: '+args.checkpoint))

    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        if epoch%10==0 and epoch>0:
            test(net,criterion,logger_test,val_loader,steps)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]

            images = Variable(images)
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]

            #prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            #acc = prec1[0]
            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1.item()  # use item() method to convert 0-dimensional tensor to scalar

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            if steps%20==0:
                print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, args.epochs, steps,
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,acc)))

            if steps%20==0:
                logger.scalar_summary('accuracy', acc, steps)
                logger.scalar_summary('loss', loss, steps)

                original = [im[0] for im in original]
                imgs = np.zeros([9,75,75,3])
                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([(im-im.min())/(im.max()-im.min())
                                         for im in img],axis=2)

                logger.image_summary('input', imgs, steps)

            steps += 1

            if steps%1000==0:
                filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
                net.save(filename)
                print('Saved: '+args.checkpoint)

            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test(net,criterion,logger,val_loader,steps):
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        #accuracy.append(prec1[0])
        accuracy.append(prec1.item())

    if logger is not None:
        logger.scalar_summary('accuracy', np.mean(accuracy), steps)
    print('TESTING: %d), Accuracy %.2f%%' %(steps,np.mean(accuracy)))
    net.train()

if __name__ == "__main__":
    main()
