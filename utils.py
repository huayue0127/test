import os
import logging
import numpy as np
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import operator
import cv2
import torch.backends.cudnn as cudnn
from metrics import *
import copy

from model import *
from dataset import *

def save_checkpoint(state, filename):
    torch.save(state, filename)

def showImg(args, pic_path, img, pred=True, fName=''):# 黑白图像
    """
    show image from given numpy image
    """
    #print('img:', img.shape)
    #print(img)
    img = img[0, 0, :, :]
    # img由四维的[1，1，560，480]变为二维数组[560，480]，即img=原img中第0个batch中第0个通道（也就是第一个batch中第一个通道的数据）中的数据
    # 降维了
    # 由于原文给出的数据集中gt为灰度图，因此直接保存模型输出图片，不做二值化
    if pred:
        img = img > args.yuzhi# 返回一个数组，img中大于0.5的数返回1，小于0.5的数返回0
    else:
        img = img > 0

# 模型通过输入数据和对应分割真值，训练，使模型输出的图像中应该被分割的部分的颜色变得更接近白色（像素对应数值更小）（因为给出的真值被分割部分为白色）

    img = Image.fromarray(np.uint8(img * 255), mode='L')
    # 归一化+L：灰度图像，由于上面将img中的数值均转换为0或1，所以这里其实相当于二值图像，即非黑即白只有0和255两种值（颜色）

    if fName:
        img.save(pic_path + fName + '.png')
        #print(pic_path)
    else:
        img.show()


def img_input(myMark_path, result_path):
    myMark = cv2.imread(myMark_path)  # 使用 cv2 来打开图片
    result = cv2.imread(result_path)  # 使用 cv2 来打开图片
    return (myMark, result)


#计算保存图像的dice系数
def dice_savepic(args, pic_path, len_test, epoch, net_id, round):
    dicelist = []
    logging.info('```````````the dice of saved test pictures````````````')
    #print('saving pictures......')
    for j in range(0, len_test):
        gt_path = pic_path + str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ":y_" + str(j) + ".png"
        pred_path = pic_path + str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ':y_pred_' + str(j) + '.png'

        gt, pred = img_input(gt_path, pred_path)

        seg_pred_flat = pred.reshape(1, -1) / 255
        gt_flat = gt.reshape(1, -1) / 255

        intersection = (seg_pred_flat * gt_flat).sum(1)
        unionset = seg_pred_flat.sum(1) + gt_flat.sum(1)  # .sum(1)求数组每一行的和，.sum(0)求数组每一列的和
        dice = (2 * intersection + args.smooth) / (unionset + args.smooth)

        dicelist.append(dice)
        #print('epoch:', str(epoch), 'The num :', j, '-dice is:', dice)
        logging.info('   epoch :{},The num :{}-dice is :{}'.format(str(epoch), j, dice))

    dicesum = 0
    num = 0
    for num in range(0, len_test):
        dicesum = dicesum + dicelist[num]

    num += 1
    avg = dicesum / num
    #print("avg-dice:", avg)
    logging.info('avg-dice:{}'.format(avg))
    logging.info('``````````````````````````````````````````````````````')


def reconstruction(args, pic_path, img, fName):
    img = img[0, 0, :, :]
    img = Image.fromarray(np.uint8(img * 255), mode='L')
    img.save(pic_path + fName + '.png')#保存成别的形式的图像？？？？


def dice_loss(args, pred, gt):
    #pred = pred.permute(0, 2, 3, 1)
    #gt = gt.permute(0, 2, 3, 1)
    dicesum = 0
    #list_w = [255, 255]
    for i in range(len(gt)):
        #print('pred[i]:', pred[i].size())

        N = gt[i].size(0)
        pred_flat = pred[i].view(N, -1)
        gt_flat = gt[i].view(N, -1)
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        dice = (2 *intersection + args.smooth) / (unionset + args.smooth)
        #print(intersection)

        #c1, count1, white1, black1 = img_size_loss(pred[i])
        #c2, count2, white2, black2 = img_size_loss(gt[i])
        #size = 0
        #dice = (2 * size + soomth) / (white1 + white2 + soomth)
        dicesum += dice

    avg = dicesum / len(gt)
    #print("avg:", avg)
    return 1-avg

def partition_data(len, logdir, client_num, cls):
    idxs = np.random.permutation(len)
    batch_idxs = np.array_split(idxs, client_num)
    net_dataidx_map = {i: batch_idxs[i] for i in range(client_num)}
    '''
    if (cls == True):
        # 划分分类数据集时同时显示每个划分中的类别数量
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return net_dataidx_map, traindata_cls_counts
    '''
    return net_dataidx_map

def init_nets(args, client_menu=None, type=None):
    if type == 'client':
        nets = {net_i: None for net_i in client_menu}
        # 使用clinet_menu中的每个元素作为字典nets中的key，通过key值判断网络类型决定初始化方式
        for net_i in client_menu:
            if net_i[-3:] == 'cls':
                net = CLS(args.useBN)
                if args.cuda:
                    net = net.cuda()
                    cudnn.benchmark = False
                    cudnn.deterministic = True
                nets[net_i] = net
            elif net_i[-3:] == 'seg':
                net = SEG(args.useBN)
                if args.cuda:
                    net = net.cuda()
                    cudnn.benchmark = False
                    cudnn.deterministic = True
                nets[net_i] = net
    elif type == 'global':
        net = Encoder(args.useBN)
        if args.cuda:
            nets = net.cuda()
            cudnn.benchmark = False
            cudnn.deterministic = True
    elif type == 'cls':
        net = CLS(args.useBN)
        if args.cuda:
            nets = net.cuda()
            cudnn.benchmark = False
            cudnn.deterministic = True
    elif type == 'seg':
        net = SEG(args.useBN)
        if args.cuda:
            nets = net.cuda()
            cudnn.benchmark = False
            cudnn.deterministic = True

    '''model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)'''

    return nets


def save_testpic(net_id, net, args, epoch, pic_path, round, test_dataloader):
    # 存图（r+c+s）
    with torch.no_grad():
        for i, (img1, gt) in enumerate(test_dataloader):
            # print('gt-h:', gt)
            if args.cuda:
                img1 = img1.cuda()
                gt = gt.cuda()
            if i == args.save_pic_num:
                break

            y_pred_seg2, _ = net(Variable(img1))
            # ————————————————————————————————————————————————————————————————————————————————————————
            # 这里分割测试集的batchsize=1，因此不用循环
            map = y_pred_seg2[0, 0, :, :].unsqueeze(0).unsqueeze(0)
            y_pred_seg = map
            # ————————————————————————————————————————————————————————————————————————————————————————

            # print('y', y.size(), 'y_pred', y_pred.size())
            # 保存原图，分割结果，真值，均保存为二值图像
            # showImg(img.cpu().numpy(), binary=True, fName=str(epoch) + ':x_ori_' + str(i))
            # print('y_pred_seg', y_pred_seg.size())
            # print('gt', gt.size())
            showImg(args, pic_path, y_pred_seg.data.cpu().numpy(), pred=True,
                        fName=str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ':y_pred_' + str(i))
            showImg(args, pic_path, gt.cpu().numpy(), pred=False,
                        fName=str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ':y_' + str(i))
            # 保存图像重建结果图片及原图x，灰度图
            '''reconstruction(args, pic_path, y_pred_re.data.cpu().numpy(),
                               fName=str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ':reconstruction' + str(i))
            reconstruction(args, pic_path, img1.cpu().numpy(),
                               fName=str(round) + ':' + str(net_id) + '(' + str(epoch) + ')' + ':original_x' + str(i))'''

    # 显示已存的测试分割图的dice系数，并计算所有分割测试集数据的dice系数
    len_test = args.save_pic_num
    dice_savepic(args, pic_path, len_test, epoch, net_id, round)

'''def save_reconstruction(net_id, net, args, epoch, pic_path, round, test_dataloader):
    for i, (img1, gt) in enumerate(test_dataloader):
        # print('gt-h:', gt)
        if args.cuda:
            img1 = img1.cuda()
            gt = gt.cuda()
        if i == args.save_pic_num:
            break
        y_pred_re, y_pred_cls = net(Variable(img1))
        reconstruction(args, pic_path, y_pred_re.data.cpu().numpy(),
                       fName='round:' + str(round) + ':' + 'id:' + str(net_id) + '_' + str(epoch) + ':reconstruction' + str(i))
        reconstruction(args, pic_path, img1.cpu().numpy(),
                       fName='round:' + str(round) + ':' + 'id:' + str(net_id) + '_' + str(epoch) + ':original_x' + str(i))'''

# 测试
def test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader):
    logging.info('[[[[[[[[[[[[[[start test]]]]]]]]]]]]]]]]]]')
    # 不计算梯度，也不反向传播
    net.eval()
    if net_id[-3:] == 'cls':
        # 在分类数据集中计算所有测试数据的：分类准确性+重建准确性
        reconsum = 0
        correct = 0
        pred_1 = 0
        pred_0 = 0
        label_1 = 0
        label_0 = 0
        count = 0
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dataloader):
                count = count + 1
                if args.cuda:
                    img = img.cuda()
                    label = label.cuda()

                # y_pred_seg, y_pred_re, y_pred_cls = model(Variable(img))
                y_pred_cls, _ = net(Variable(img))
                N = img.size(0)  # y和y_pred均为[1,1,560,480],.size（0）为获得y第一个维度的大小，所以y.size(0)=1，N=1
                img_flat = img.view(N, -1)

                pred_cls = y_pred_cls.max(1, keepdim=True)[1]  # 值，索引
                # pred = torch.max(output, dim=1)
                # pred = output.argmax(dim=1)
                # 累计正确的值
                correct += pred_cls.eq(label.view_as(pred_cls)).sum().item()
                pred_list = pred_cls.view_as(
                    label).cpu().numpy().tolist()  # 由[[1],[0],...,[1]]——>变为[1,0,...,1],因为label就是后者的形式
                label_list = label.cpu().numpy().tolist()
                # print('pred_list', pred_list)
                # print('label_list', label_list)
                pred_1 += pred_list.count(1)
                pred_0 += pred_list.count(0)
                label_1 += label_list.count(1)
                label_0 += label_list.count(0)

            total_used_pics = count * args.test_batchsize
            accuracy = correct / total_used_pics
            TP = (correct + label_1 - pred_0) / 2
            Precision = TP / (pred_1 + args.smooth)
            Recall = TP / (label_1 + args.smooth)
            # logging.info('【【net_id:{}'.format(net_id))
            logging.info('Test_{} pics(the average metrics of all test data are shown below)'.format(total_used_pics))
            logging.info(
                '{}-classification accuracy:{}, Precision:{}, Recall:{}'.format(net_id, accuracy, Precision, Recall))

    if net_id[-3:] == 'seg':
        # 在分割数据集中计算所有测试数据的：分割dice系数的均值及方差
        reconsum = 0
        dice_list = []
        dicesum = []
        HD_list = []
        HD_sum = []
        HD_95_list = []
        HD_95_sum = []
        IOU_list = []
        IOU_sum = []
        ASSD_list = []
        ASSD_sum = []
        ASD_list = []
        ASD_sum = []
        count = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dataloader):
                count = count + 1
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()

                #y_pred_re, y_pred_seg1 = net(Variable(x))
                NET = copy.deepcopy(net)
                y_pred_seg1, _ = net(Variable(x))

                # ————————————————————————————————————————————————————————————————————————————————————————
                map = y_pred_seg1[0, 0, :, :].unsqueeze(0).unsqueeze(0)
                # [1,400,560]
                # softmax输出的两个通道中只选一个
                for j in range(1, args.test_batchsize):  # 4为分割测试集的batchsize
                    m = y_pred_seg1[j, 0, :, :]
                    m = m.unsqueeze(0).unsqueeze(0)
                    map = torch.cat((map, m), 0)
                y_pred_seg = map
                # logging.info(y_pred_seg)
                # print(y_pred_seg.size())
                # print(y.size())
                # ————————————————————————————————————————————————————————————————————————————————————————

                # python中[2,3]应理解为一个数组由两个有三个元素的一维数组组成，【[1,2,3][4,5,6]】
                N = y.size(0)  # y和y_pred均为[1,1,560,480],.size（0）为获得y第一个维度的大小，所以y.size(0)=1，N=1

                # 先将网络输出的结果二值化
                y_pred_seg = y_pred_seg.cpu().numpy() > args.yuzhi
                y = y.cpu().numpy()  # gt经过数据读取已经是二值化的了
                # Dice
                dice = Dice(args, y_pred_seg, y)  # dice大小为[1, batchsize]
                dice_list.append(dice)  # dice_list为：[[dice1], [dice2], ...., [dice3]]
                dicesum = dicesum + dice  # dicesum是由每张图的dice值组成的列表，这里的+是指在dicesum列表后面链接dice列表。在最后求值的时候需要把所有元素相加

                # IOU
                IOU = IOU_Jaccard(y_pred_seg, y)
                IOU_list.append(IOU)
                IOU_sum = IOU_sum + IOU

                # HD
                HD = torch2D_Hausdorff_distance(y_pred_seg, y, voxelspacing=args.voxelspacing)  # 大小应为[1, batchsize]
                HD_list.append(HD)
                HD_sum = HD_sum + HD

                # HD_95
                HD_95 = binary_hausdorff95(y_pred_seg, y, voxelspacing=args.voxelspacing)  # 大小应为[1, batchsize]
                HD_95_list.append(HD_95)
                HD_95_sum = HD_95_sum + HD_95

                # ASSD
                ASSD = ASSD_binary_assd(y_pred_seg, y)
                ASSD_list.append(ASSD)
                ASSD_sum = ASSD_sum + ASSD

                # ASD
                ASD = ASD_binary_asd(y_pred_seg, y, voxelspacing=args.voxelspacing)
                ASD_list.append(ASD)
                ASD_sum = ASD_sum + ASD

            total_used_pics = count * args.test_batchsize
            dice_avg = sum(dicesum) / total_used_pics
            dice_var = np.var(dice_list)
            HD_avg = sum(HD_sum) / total_used_pics
            HD_var = np.var(HD_list)
            HD_95_avg = sum(HD_95_sum) / total_used_pics
            HD_95_var = np.var(HD_95_list)
            IOU_avg = sum(IOU_sum) / total_used_pics
            IOU_var = np.var(IOU_list)
            ASSD_avg = sum(ASSD_sum) / total_used_pics
            ASSD_var = np.var(ASSD_list)
            ASD_avg = sum(ASD_sum) / total_used_pics
            ASD_var = np.var(ASD_list)
            # logging.info('net_id:{}'.format(net_id))
            # logging.info('【【net_id:{}'.format(net_id))
            logging.info('Test:{} pics(the average metrics of all test data are shown below)'.format(total_used_pics))
            logging.info('{}-dice:{}, (dice-var:{})'.format(net_id, dice_avg, dice_var))
            logging.info('{}-JC(IOU):{}, (JC-var:{})'.format(net_id, IOU_avg, IOU_var))
            logging.info('{}-HD:{}, (HD-var:{})'.format(net_id, HD_avg, HD_var))
            logging.info('{}-HD_95:{}, (HD_95-var:{})'.format(net_id, HD_95_avg, HD_95_var))
            logging.info('{}-ASD:{}, (ASD-var:{})'.format(net_id, ASD_avg, ASD_var))
            logging.info('{}-ASSD:{}, (ASSD-var:{})'.format(net_id, ASSD_avg, ASSD_var))
            # 保存客户端测试结果图，并计算dice
            save_testpic(net_id, net, args, epoch, pic_path, round, test_dataloader)
            # 之前这里作为测试的一部分，需要通过网络训练得到结果，但是不在with no_grad的下面！！所以会计算梯度？？
            # 而且net作为一个字典，在函数中对他的值进行了更改，实参的值也会随之更改？？
    net.train()
    logging.info('[[[[[[[[[[[[end the test]]]]]]]]]]]]]]]]]')


# 每个客户端进行本地训练
def train_net(args, round, net_id, net, train_dataloader, test_dataloader, pic_path, global_model, cls_model=None, seg_model=None, regular=False):
    #net = nn.DataParallel(net)
    net.cuda()
    net.train()
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    #logging.info('n_test: %d' % len(test_dataloader))
    #train_acc,_ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    #logging.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    #logging.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    #optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    
    if regular == False:
        epoch = 0
        test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)
    if regular == False:
        train_epochs = args.epochs
    elif regular == True:
        train_epochs = args.regular_epochs

    if net_id[-3:] == 'cls':
        for epoch in range(train_epochs):
            count = 0
            loss_cls = nn.CrossEntropyLoss()
            loss_r_MSE = nn.MSELoss()
            lossfunc_KL = nn.KLDivLoss()
            if args.cuda:
                loss_cls = loss_cls.cuda()
                lossfunc_KL = lossfunc_KL.cuda()
            loss_c_sum = 0  # 图像分类损失

            for i, (img, label) in enumerate(train_dataloader):
                count = count + 1
                img, label = Variable(img), Variable(label)
                if args.cuda:
                    img = img.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                img.requires_grad = False
                label.requires_grad = False
                # 分别输出图像分类和图像重建的结果
                y_pred_class, before_sfm = net(img)
                loss_c = loss_cls(y_pred_class, label.long())
                proximal_term = 0
                if regular == True:
                    _, before_sfm_cls = cls_model(img)
                    loss_KL = lossfunc_KL(F.log_softmax(before_sfm, dim=1), F.softmax(before_sfm_cls, dim=1))
                    loss_c = loss_c + loss_KL

                    '''co = 0
                    for w, w_cls in zip(net.parameters(), cls_model.parameters()):
                        co = co + 1
                        proximal_term = proximal_term + (w - w_cls).norm(2)
                        #print(proximal_term)
                    print('count:', co)
                    loss_rc = loss_rc + proximal_term'''
                loss_c.backward()
                optimizer.step()
                loss_c_sum += loss_c
            total_used_pics = count*args.batchSize
            loss_c_sum = loss_c_sum.item()/total_used_pics
            print('{}__epoch: {}(round:{}), loss_c:{}'.format(net_id, epoch, round, loss_c_sum))
            logging.info('{}__epoch: {}(round:{}), loss_c:{}'.format(net_id, epoch, round, loss_c_sum))

            if epoch % args.saveround == 0 and regular == False:
                #with torch.no_grad():
                test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)
        logging.info(' *** round:{}--{} Training complete ***'.format(round, net_id))
        print(' *** round:{}--{} Training complete ***'.format(round, net_id))

    if net_id[-3:] == 'seg':
        loss_MSE = nn.MSELoss()
        loss_r_MSE = nn.MSELoss()
        lossfunc_KL = nn.KLDivLoss()
        if args.cuda:
            loss_MSE = loss_MSE.cuda()
            loss_r_MSE = loss_r_MSE.cuda()
            lossfunc_KL = lossfunc_KL.cuda()

        for epoch in range(train_epochs):
            count = 0
            loss_s_sum = 0  # 重建+分类总loss
            loss_s_MSE_sum = 0  # 图像分割的总均方差损失
            loss_s_dice_sum = 0  # 图像分割的总diceloss
            #test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)
            for i, (img, gt) in enumerate(train_dataloader):
                count = count + 1
                img, gt = Variable(img), Variable(gt)
                if args.cuda:
                    img = img.cuda()
                    gt = gt.cuda()

                '''if i == 0:
                    test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)'''
                img.requires_grad = False
                gt.requires_grad = False
                # 第一个batch更新前本地客户端的网络参数=全局模型网络参数（因为只有一个客户端，所以就应该是这样）
                # 之后的batch不一样了——就应该是这样
                NET = copy.deepcopy(net)
                y_pred_seg1, before_sfm = net(img)

                # ————————————————————————————————————————————————————————————————————————————————————————
                map = y_pred_seg1[0, 0, :, :].unsqueeze(0).unsqueeze(0)
                # [1,400,560]
                # softmax输出的两个通道中只选一个
                for j in range(1, args.batchSize):
                    m = y_pred_seg1[j, 0, :, :]
                    m = m.unsqueeze(0).unsqueeze(0)
                    map = torch.cat((map, m), 0)
                y_pred_seg = map
                # logging.info(y_pred_seg)
                # print(y_pred_seg.size())
                # ————————————————————————————————————————————————————————————————————————————————————————
                loss_s_MSE = loss_MSE(y_pred_seg, gt)  # 平方差损失
                loss_s_dice = dice_loss(args, y_pred_seg, gt)  # 图像分割的diceloss
                '''print(loss_s_MSE.requires_grad)
                print(loss_r.requires_grad)
                print(loss_s_dice.requires_grad)'''
                # print('label', label.long().type(), y_pred_class.long().type())
                loss_s = loss_s_MSE + loss_s_dice
                proximal_term = 0
                if regular == True:
                    _, before_sfm_seg = seg_model(img)
                    loss_KL = lossfunc_KL(F.log_softmax(before_sfm, dim=1), F.softmax(before_sfm_seg, dim=1))
                    loss_s = loss_s + loss_KL
                    '''for w, w_seg in zip(net.parameters(), seg_model.parameters()):
                        proximal_term = proximal_term + (w - w_seg).norm(2)
                    loss_rs = loss_rs + proximal_term'''
                #print(loss_rs.requires_grad)——True

                optimizer.zero_grad()
                loss_s.backward()# 别加重建了，只是分割，看看结果
                optimizer.step()
                loss_s_MSE_sum = loss_s_MSE_sum + loss_s_MSE
                loss_s_dice_sum = loss_s_dice_sum + loss_s_dice
                loss_s_sum = loss_s_MSE_sum + loss_s_dice_sum

            total_used_pics = count * args.batchSize
            loss_s_sum = loss_s_sum.item() / total_used_pics
            loss_s_MSE_sum = loss_s_MSE_sum.item() / total_used_pics
            loss_s_dice_sum = loss_s_dice_sum.item() / total_used_pics
            print('{}__epoch: {}(round:{}), loss_sum: {}, loss_s_MSE:{}, loss_s_dice:{}'.format(net_id, epoch, round, loss_s_sum, loss_s_MSE_sum, loss_s_dice_sum))
            logging.info('{}__epoch: {}(round:{}), loss_sum: {}, loss_s_MSE:{}, loss_s_dice:{}'.format(net_id, epoch, round, loss_s_sum, loss_s_MSE_sum, loss_s_dice_sum))

            if epoch % args.saveround == 0 and regular == False:
                #with torch.no_grad():
                #NET = copy.deepcopy(net)
                test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)

        #net.to('cpu')
    if regular == True:
        print('----after regular result:-----------------------------------------')
    test_fed(net_id, net, epoch, args, pic_path, round, test_dataloader)

    return net
    # 总loss（各loss简单相加）、图像分类、图像重建、图像分割平方差loss、图像分割diceloss
    # print('y', y_true.size(), 'y_pred', y_pred.size(), 'len', len(y_pred))
    #return train_acc, test_acc

def make_dataset(args, client_menu):
    # ---数据集加载---
    # 将进行分类任务的客户端的数据集放在字典中，key=客户端名=对应数据集的文件夹名。因为现在一个客户端只做一种任务，因此不用区分一个客户端中的数据集了，一个客户端一个数据集
    train_dataset = {client: None for client in client_menu}
    test_dataset = {client: None for client in client_menu}
    for client in client_menu:
        dataroot = os.path.join(args.dataroot, client)
        # print(dataroot)
        if client[-3:] == 'cls':
            train_dataset[client] = classification_dataset(dataroot, train=True)
            test_dataset[client] = classification_dataset(dataroot, train=False)
        elif client[-3:] == 'seg':
            train_dataset[client] = segmentation_dataset(dataroot, train=True)
            test_dataset[client] = segmentation_dataset(dataroot, train=False)
    return train_dataset, test_dataset

def count_dataset(args, nets,train_dataset, client_menu):
    data_num = {client: None for client in client_menu}

    for net_id, net in nets.items():
        train_dataset_net_id = train_dataset[net_id]
        train_dataset_net_id_len = len(train_dataset_net_id)
        truly_used = train_dataset_net_id_len - (
                    train_dataset_net_id_len % args.batchSize)  # 只有在droplast=True时才有，否则实际使用数据量=全部数据
        # 因为在生成dataloader时，有droplast，因此实际使用的数据量 <= 存储的数据量.在计算loss时需要使用到实际使用的数据量以更准确地计算loss的平均值
        if net_id[-3:] == 'cls':
            logging.info('n_training_classification: Total data volume-{} pics. Actual used-{} pics.'.format(
                train_dataset_net_id_len, truly_used))
            data_num[net_id] = truly_used
        elif net_id[-3:] == 'seg':
            logging.info('n_training_segmentation: Total data volume-{} pairs. Actual used-{} pics.'.format(
                train_dataset_net_id_len, truly_used))
            data_num[net_id] = truly_used

    return data_num

#所有的客户端进行本地训练
def local_train_net(args, round, nets, pic_path, client_menu, global_model, train_dataset, test_dataset, cls_model=None, seg_model=None, regular=False):
    # 对每个客户端模型进行训练
    # net_id即为client_menu中的名字

    # train_dataset = {'客户端名/数据集文件夹名'：对应训练数据集}
    data_num = count_dataset(args, nets, train_dataset, client_menu)
    #data_num = {client: None for client in client_menu}

    for net_id, net in nets.items():
        # dl:dataloader
        # train_dataset:值为所有客户端训练数据的字典，train_dataset[net_id]为当前客户端数据集，不打乱顺序，因为要用随机种子
        train_dataset_net_id = train_dataset[net_id]
        test_dataset_net_id = test_dataset[net_id]
        # 用当前客户端的数据集train_dataset_net_id，获取当前数据集的dataloader：train_dl_net_id
        train_dl_net_id = torch.utils.data.DataLoader(train_dataset_net_id, batch_size=args.batchSize, num_workers=args.workers,
                                                      shuffle=False, pin_memory=False, drop_last=True)
        test_dl_net_id = torch.utils.data.DataLoader(test_dataset_net_id, batch_size=args.test_batchsize, num_workers=args.workers,
                                                      shuffle=False, pin_memory=False, drop_last=True)

        logging.info('>>>>>>>>>>>>>>>>>>>>Training network %s<<<<<<<<<<<<<<<<<<<<<<<<<<' % str(net_id))

        # 客户端net_id进行本地训练
        #optimizer_net_id = optimizer_all[net_id]
        #print(optimizer_net_id)
        net = train_net(args, round, net_id, net, train_dl_net_id, test_dl_net_id, pic_path, global_model, cls_model, seg_model, regular)
        logging.info(' <<<<<<<<<<<<<<<<<round:{}--{} Training complete >>>>>>>>>>>>>>>>>>>>>>'.format(round, net_id))
        print('<<<<<<<<<<<<<<<<<round:{}--{} Training complete >>>>>>>>>>>>>>>>>>>>>>'.format(round, net_id))


    return data_num, nets
