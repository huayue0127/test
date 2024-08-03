import numpy as np
import torch
from dataset import *
from model import Net
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
#import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from torch.autograd import Variable
import shutil
import json
import os
import cv2
import operator
import matplotlib.pyplot as plt
import os
import datetime
import logging
from tqdm import tqdm
import pytz
import random
import test
import argparse
import copy

from utils import *

'''
所有客户端中均有重建任务
默认前几个客户端是分类+重建，后几个是分割+重建：0~n-1是分类，n~n_parties-1是分割
当前数据集可对应6个客户端，即认为是现实中六个不同的机构
round:也被称为comm_round，表示通信轮次。niter：也称为n_comm_rounds,服务器和客户端一共进行多少轮通信
epoch:表示在每两轮通信之间，每个客户端的更新轮次。epochs：为每个客户端在两轮通信之间一共需要进行进行几次本地更新
将各客户端的数据集按cls_data_menu、seg_data_menu中的顺序分别放在字典cls_train_dataset、seg_train_dataset中，key=客户端名=对应数据集的文件夹名
进行不同任务的客户端和他们对应的数据集，分别以key-value的形式存放在统一的字典nets和train_dataset中。
train_dataset中的每个value均为数据集对象
dl:dataloader
当前全局模型仅有encoder，不是一个完整的模型，因此不需要对全局模型进行测试，因此不需要相应的数据集，因此数据集不需要有全局和局部之分，只有客户端数据集了
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/18942690292/huayue/data/fed_data', help='path to dataset of kaggle ultrasound nerve segmentation')
#parser.add_argument('--testroot', default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)###
parser.add_argument('--batchSize', type=int, default=64, help='input train batch size')
parser.add_argument('--test_batchsize', type=int, default=4, help='input test data batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')# n_comm_rounds,服务器和客户端一共通信多少轮
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train for in client')# 每个客户端在两轮通信之间训练多少轮
#parser.add_argument('--start_round', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
#parser.add_argument('--saved-model_name', default='checkpoint___.tar', type=str, help='output checkpoint filename')# 在名字中加上时间以区分保存的不同模型
parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
parser.add_argument('--picdir', type=str, required=False, default="./pic/", help='picture path')
parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
parser.add_argument('--pic_file_name', type=str, default=None, help='The log file name')
parser.add_argument('--saveround', type=int, default=5, help='number of epochs to save pictuers')#每隔几轮存一次图
parser.add_argument('--save_pic_num', type=int, default=3, help='number of pictures will be saved after saveround')#每次存多少张图
parser.add_argument('--smooth', type=float, default=1e-5, help='the smooth of dice')#防止dice出现零除以零
parser.add_argument('--loss2_w', type=int, default=1, help='the weight of loss2')
parser.add_argument('--loss3_w', type=int, default=1, help='the weight of loss3')
parser.add_argument('--yuzhi', type=float, default=0.5, help='yuzhi to 2zhi pic')
parser.add_argument('--n_parties', type=int, default=6, help='number of workers in a distributed cluster')# 客户端数量
#parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')# 通信轮次
parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
parser.add_argument('--voxelspacing', type=float, default=None, help='the parameter of hd,hd95,asd,assd')
#parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')#每轮被选中参与训练的客户端的数量
parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")#权重衰减系数
#parser.add_argument('--client_menu', type=str, default=None, help='The log file name')

args = parser.parse_args()
print(args)
# 分类数据存放情况：列表中的每个值代表一个分类文件夹的名字，分别为来自不同机构的数据
cls_data_menu = ['0_cls', '1_cls', '2_cls']
# 分割数据情况，和分类数据同理
seg_data_menu = ['3_seg', '4_seg', '5_seg']
# 各客户端的名字，此顺序即为客户端顺序和对应数据集名字
#client_menu = ['0_cls', '1_cls', '2_cls']
#client_menu = ['3_seg', '4_seg', '5_seg']
client_menu = ['0_cls', '1_cls', '2_cls', '3_seg', '4_seg', '5_seg']
#client_menu = ['5_seg', '3_seg', '4_seg', '0_cls', '1_cls', '2_cls']
#client_menu = ['3_seg', '4_seg']
#client_menu = ['4_seg']
#client_menu = ['1_cls']
'''
def beijing(sec, what):#修正当前时间
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=7)
    return beijing_time.timetuple()
    '''

if __name__ == '__main__':
    # 先判断args.n_parties和client_menu的长度是否相等，因为后面要经常用到。必须是相等的，否则报错并退出程序
    if args.n_parties != len(client_menu):
        print('the length of client_menu is not equal to args.n_parties, please revise and run again!!')
        exit(0)
    tz = pytz.timezone('Asia/Shanghai')#不知道为什么突然变成波兰时区了,改为中国时区
    cn_time = datetime.datetime.now(tz)#修正文件名时间
    #logging.Formatter.converter = beijing#修正log文件中的时间
    '''
    >>> import time
    >>> print(time.tzname())

    '''

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.picdir, exist_ok=True)

    print(datetime.datetime.now(tz).strftime("%Y-%m-%d-%H%M-%S"))

    # 生成json文件
    if args.log_file_name is None:
        argument_path = 'arguments-%s.json' % cn_time.strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    # 生成log文件
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (cn_time.strftime("%Y-%m-%d-%H%M-%S"))

    log_path = args.log_file_name + '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path),
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', filemode='w')

    logger = logging.getLogger()  # 创建log
    logger.setLevel(logging.INFO)  # 设置log的等级，info等级低于debug，所以才能用info向debug等级中写入
    logging.info(args)
    # logger.info(device)

    if args.pic_file_name is None:
        args.pic_file_name = 'pic-%s' % (cn_time.strftime("%Y-%m-%d-%H%M-%S"))
    # args.pic_file_name = args.pic_file_name + '/'
    pic_path = os.path.join(args.picdir, args.pic_file_name, '')  # 最后没有/，所以想要在后续中作为地址存东西，需要添加/
    os.makedirs(pic_path, exist_ok=True)

    # 随机种子
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现

    # ############# dataset processing
    # 分割训练数据集
    # dataset = kaggle2016nerve(args.dataroot, train=True)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, num_workers=args.workers, shuffle=True, pin_memory=True)
    # workers:多线程读取数据，dataloder自主加载数据到RAM中，每个worker对应自主加载一个batch的数据
    # 使每一轮迭代时dataloder在RAM中寻得本轮要使用的batch的速度加快，加快程序巡行速度
    # pin_memory:内存寄存，在数据返回前，将数据复制到cuda内存中

    '''# ---数据集加载---
    # 将进行分类任务的客户端的数据集放在字典中，key=客户端名=对应数据集的文件夹名。因为现在一个客户端只做一种任务，因此不用区分一个客户端中的数据集了，一个客户端一个数据集
    train_dataset = {client: None for client in client_menu}
    test_dataset = {client: None for client in client_menu}
    for client in client_menu:
        dataroot = os.path.join(args.dataroot, client)
        #print(dataroot)
        if client[-3:] == 'cls':
            train_dataset[client] = classification_dataset(dataroot, train=True)
            test_dataset[client] = classification_dataset(dataroot, train=False)
        elif client[-3:] == 'seg':
            train_dataset[client] = segmentation_dataset(dataroot, train=True)
            test_dataset[client] = segmentation_dataset(dataroot, train=False)
    # train_dataset = {'客户端名/数据集文件夹名'：对应训练数据集}'''


    # 分割分类，分割数据集——分割情况用下标表示即分割下标
    #net_dataidx_map_cls = partition_data(len(cls_train_dataset), args.logdir, args.n_parties, cls=True)
    #net_dataidx_map_seg = partition_data(len(seg_train_dataset), args.logdir, args.n_parties, cls=False)
    #n_party_per_round = int(args.n_parties * args.sample_fraction)# 每轮几个客户端参与本地训练

    #party_list = [i for i in range(args.n_parties)]
    # 所有客户端都参与每轮的通信
    '''
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(n_comm_rounds):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(n_comm_rounds):
            party_list_rounds.append(party_list)
            '''

    # 网络初始化
    logging.info("Initializing nets......")
    nets = init_nets(args, client_menu=client_menu, type='client')
    global_model = init_nets(args, type='global')# 全局模型不需要具体的网络，只用字典变量作为全局模型？？
    global_weight = global_model.state_dict()
    
    #last_model = init_nets(args, type='global')#保存客户端刚接收到的服务器模型
    #global_model = global_models[0]
    #optimizer = optim.Adagrad(nets.parameters(), lr=args.lr)

    # ---优化器加载，加载到一个字典里:optimizer_all,后面用optimizer表示每个客户端的优化器
    optimizer_all = {client: None for client in client_menu}
    for client in optimizer_all.keys():
        net = nets[client]
        optimizer_all[client] = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        #print(optimizer_dic[client])

    # 设定一些参数的初始值
    round = 0
    n_comm_rounds = args.niter  # 通信轮次

    ############## resume:继续运行保存好的模型-----------------------
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading saved model '{}'".format(args.resume))
            logging.info("=> loading saved model '{}'".format(args.resume))

            if args.cuda == False:
                saved_model = torch.load(args.resume, map_location={'cuda:0': 'cpu'})
            else:
                saved_model = torch.load(args.resume, map_location={'cuda:0': 'cuda'})

            # 加载优化器参数
            optimizer.load_state_dict(saved_model['optimizer'])

            global_model.load_state_dict(saved_model['global_model'])
            print('loaded global model')
            nets.load_state_dict(saved_model['client_model'])
            print('loaded client model')
            round = saved_model['comm_round'] #因为在保存模型时已经给round+1了，因此运行保存模型时，round直接从加载的comm_round开始统计轮次
            print("=> loaded checkpoint (comm_round:{})".format(saved_model['comm_round']))
            logging.info("=> loaded checkpoint (comm_round:{})".format(saved_model['comm_round']))
            logging.info('loaded saved global model, client model and optimizer ')

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


# 客户端和服务器之间地通信轮次为：round，每个客户端的训练轮次为：epoch
#range(round, args.niter):如果加载保存好的模型，则从加载的round开始计数，如果没有加载模型，则round=0，从0开始计数
    for round in tqdm(range(round, args.niter)):# n_comm_rounds = args.niter
        logging.info("===========in comm round:" + str(round) + "==============================================")
        print("=================in comm round:" + str(round) + "====================================")

        # 本地训练
        logging.info('-----local model train---------------------------------')
        data_num, nets = local_train_net(args, round, nets, pic_path, client_menu, global_model, optimizer_all)
        '''net_para = net.state_dict()
        key = 'Conv1.0.weight'
        print('after-key:', net_para[key])'''
        # 返回值data_num为一个以各client为key，以实际使用的数据量为value的字典
        # 进行服务器与客户端之间的通信,在服务器进行模型聚合
        logging.info('-----global model aggregation--------------------------')
        total_data_points = sum([num for num in data_num.values()])
        fed_avg_freqs = {client: None for client in client_menu}
        for client_id, num in data_num.items():
            fed_avg_freqs[client_id] = torch.tensor(num / total_data_points, dtype=torch.double)
            #fed_avg_freqs[client_id] = torch.tensor(1, dtype=torch.double)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print('fed_avg_freqs', fed_avg_freqs)

        for net_id, net in nets.items():
            net_para = net.state_dict()# 由网络层名称+参数——组成的字典
            first = client_menu[0]
            #print(first)
            first_index = first[0]
            #print(first_index)
            if net_id[0] == first_index:# first_index为了保证在client——menu中无论哪个在前时都能正常进行
                for key in net_para:# 只聚合client中的encoder
                    #logging.info('key:{}'.format(key))
                    if ('Conv1' in key) or ('Conv2' in key) or ('Conv3' in key) or ('Conv4' in key) or ('Conv5' in key):
                        #print('some key:', key)
                        #print(global_weight[key])
                        global_weight[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    if ('Conv1' in key) or ('Conv2' in key) or ('Conv3' in key) or ('Conv4' in key) or ('Conv5' in key):
                        global_weight[key] = global_weight[key] + net_para[key] * fed_avg_freqs[net_id]

        global_model.load_state_dict(global_weight) #感觉都不需要全局模型这个模型了，只需要一个可以存放聚合后的encoder参数的变量

        for net in nets.values():
            net.load_state_dict(global_weight, strict=False)
            '''net_para = net.state_dict()
            key = 'Conv1.0.weight'
            net_para[key] = net_para[key].cpu().numpy().tolist()
            print('after-key:', net_para[key][0])
            logging.info('after-key:{}'.format(net_para[key]))'''

        '''count = 0
        a = 0
        b = 0
        c = 0
        for key in net_para:
            if ('Conv1' in key) or ('Conv2' in key) or ('Conv3' in key) or ('Conv4' in key) or ('Conv5' in key):
                #print(key)
                count = count + 1
                #logging.info(net_para[key].cpu().numpy().tolist())
                netpara = net_para[key].cpu().numpy().tolist()
                #net.load_state_dict(global_weight, strict=False)
                after_para_net = net.state_dict()[key].cpu().numpy().tolist()
                if after_para_net == netpara:
                    a = a + 1
                    #exit(0)
                    #exit(0)
        print(count)#正常应该都是70,encoder确实是有70个参数
        print(a)'''
            # 看看加载完之后一不一样————————是一样的，一个客户端的话：本地网络=全局网络=加载后的本地网络

        saved_model_name = 'checkpoint-%s.tar' % (cn_time.strftime("%Y-%m-%d-%H%M"))
        # 由于还没想好怎么保存各个客户端的联邦模型，所以先不进行模型保存这一步了
        #saved_state = {'comm_round': round + 1, 'global_model': global_model.state_dict(), 'client_model': nets.state_dict(), 'optimizer': optimizer.state_dict()}
        #save_checkpoint(saved_state, saved_model_name)

        '''
        # 只对全局模型在测试集上的表现进行测试
        with torch.no_grad():
            test_fed(net_id, net, round, args, pic_path, type='global', round=round)
        '''
        logging.info("===========comm round:" + str(round) + "have done==============================================")
        logging.info(' ')
        logging.info(' ')
