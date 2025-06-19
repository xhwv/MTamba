import argparse
import random
import logging
import time
import setproctitle
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.MFEM import DMEM
from models.cssmf import *
from data.preprocess import BraTS
from torch.utils.data import DataLoader
from models.SegMamba import SegMamba
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score,accuracy_score
from tensorboardX import SummaryWriter
from torch import nn
from ranger import Ranger


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='UCSF', type=str)

parser.add_argument('--experiment', default='TransBTS_Boundary', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS_Boundary,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/home/mamba_idh_grade', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

parser.add_argument('--dataset', default='UCSF', type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)#softmax_dice

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='2', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=2, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=2000, type=int)

parser.add_argument('--save_freq', default=250, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()

class MultiTaskLossWrapper_OS(nn.Module):
    def __init__(self, ):
        super(MultiTaskLossWrapper_OS, self).__init__()
        self.vars = nn.Parameter(torch.tensor((1.0,1.0,1.0,1.0,1.0,1.0),requires_grad=True)) #1.0, 6.0
    def forward(self, loss1,loss2,loss3,loss4,loss5,loss6):
        lossd_1 = torch.sum(0.5 * loss1 / (self.vars[0] ** 2) + torch.log(self.vars[0]), -1)
        lossd_2 = torch.sum(0.5 * loss2 / (self.vars[1] ** 2) + torch.log(self.vars[1]), -1)
        lossd_3 = torch.sum(0.5 * loss3 / (self.vars[2] ** 2) + torch.log(self.vars[2]), -1)
        lossd_4 = torch.sum(0.5 * loss4 / (self.vars[3] ** 2) + torch.log(self.vars[3]), -1)
        lossd_idh=torch.sum(0.5 * loss5 / (self.vars[4] ** 2) + torch.log(self.vars[4]), -1)
        lossd_grade = torch.sum(0.5 * loss6 / (self.vars[5] ** 2) + torch.log(self.vars[5]), -1)
        loss = torch.mean(lossd_1+lossd_2+lossd_3+lossd_4+lossd_idh+lossd_grade)
        return loss
def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    weights1 = torch.ones_like(target).float()
    weights1[target == 1] = 2.0
    weights2 = torch.ones_like(target).float()
    weights2[target == 2] = 2.0
    weights3 = torch.ones_like(target).float()
    weights3[target == 4] = 2.0

    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float(),weight=weights1)
    loss2 = Dice(output[:, 2, ...], (target == 2).float(),weight=weights2)
    loss3 = Dice(output[:, 3, ...], (target == 4).float(),weight=weights3)

    return loss0,loss1,loss2,loss3

def main_worker():

    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    model = SegMamba().float().cuda()
    cssmf=CSSMF()
    idh=IDH_Predict()
    grade=Grade_Predict()
    t2flairmodel = DMEM()
    MTL = MultiTaskLossWrapper_OS().float().cuda()
    nets = {
        'model': model.cuda(),
        't2f':t2flairmodel.cuda(),
        'mtl': MTL.cuda(),
        'fusion':cssmf.cuda(),
        'idh':idh.cuda(),
        'grade':grade.cuda(),
    }
    param = [p for v in nets.values() for p in list(v.parameters()) if
             p.requires_grad]  # Only parameters that require gradients
    optimizer = Ranger(
        param,  # 网络的可训练参数
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay  # 权重衰减
    )
    criterion = softmax_dice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight1 = torch.tensor([1.0, 3.97], device=device)
    weight2 = torch.tensor([4.45, 1.0], device=device)
    # criterion1=nn.CrossEntropyLoss(weight=weight1)
    # criterion2 = nn.CrossEntropyLoss(weight=weight2)
    criterion1 = FocalLoss(weight=weight1)
    criterion2 = FocalLoss(weight=weight2)
    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    resume = ''
    writer = SummaryWriter()
    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode)
    logging.info('Samples for train = {}'.format(len(train_set)))
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    val_root = os.path.join(args.root, args.valid_dir)
    val_set = BraTS(val_list, val_root, 'valid')
    logging.info('Samples for val = {}'.format(len(val_set)))
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()
    torch.set_grad_enabled(True)
    # 假设你的其他代码保持不变
    if args.local_rank == 0:
        roott = "/home/model"
        checkpoint_dir = os.path.join(roott, 'checkpoint', args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    labels = [0, 1]  # 假设是二分类问题
    idh_names = ['IDH野生型', 'IDH突变型']
    grade_names = ['LGG', 'HGG']
    best_dice = 0.0
    best_idh_auc = 0.0
    best_grade_auc = 0.0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        print(optimizer.param_groups[0]['lr'])
        train_loss = []
        ucsf=[]
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        nets['model'].train()
        nets['t2f'].train()
        nets['fusion'].train()
        nets['idh'].train()
        nets['grade'].train()
        nets['mtl'].train()
        idh_pred_scores=[]
        grade_pred_scores=[]
        idh_preds=[]
        grade_preds=[]
        idh_trues=[]
        grade_trues=[]
        wt_dices, tc_dices, et_dices = [], [], []
        dice_0s=[]
        dice_1s=[]
        dice_2s=[]
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            x, target, idh, grade, name = data
            x = x.cuda(args.local_rank, non_blocking=True).float()
            target = target.cuda(args.local_rank, non_blocking=True)
            idh=idh.cuda(args.local_rank, non_blocking=True)
            grade=grade.cuda(args.local_rank, non_blocking=True)
            feature_en, seg_output = nets['model'](x)
            segloss0, segloss1, segloss2, segloss3 = criterion(seg_output, target)
            pred_labels = torch.argmax(seg_output, dim=1)
            mapping = {3: 4}
            for k, v in mapping.items():
                pred_labels[pred_labels == k] = v
            for idx in range(pred_labels.shape[0]):
                dice_2=dice_coeff(pred_labels[idx], target[idx], [2])
                dice_1 = dice_coeff(pred_labels[idx], target[idx], [1])
                dice_0 = dice_coeff(pred_labels[idx], target[idx], [0])
                wt_dice = dice_coeff(pred_labels[idx], target[idx], [1, 2, 4])
                tc_dice = dice_coeff(pred_labels[idx], target[idx], [1, 4])
                et_dice = dice_coeff(pred_labels[idx], target[idx], [4])
                if wt_dice==0 and tc_dice==0 and et_dice==0:
                    ucsf.append(name[idx])
                dice_2s.append(dice_2.item())
                dice_1s.append(dice_1.item())
                dice_0s.append(dice_0.item())
                wt_dices.append(wt_dice.item())
                tc_dices.append(tc_dice.item())
                et_dices.append(et_dice.item())

            flair_image = x[:, 0, :, :, :].unsqueeze(1)
            t2_image = x[:, 3, :, :, :].unsqueeze(1)
            featuret2f = nets['t2f'](t2_image, flair_image)
            output = nets['fusion'](feature_en,featuret2f)
            idh_p=nets['idh'](output)
            grade_p=nets['grade'](output)
            idh_pred_score = F.softmax(idh_p, dim=1)[:, 1].detach().cpu().numpy()
            grade_pred_score = F.softmax(grade_p, dim=1)[:, 1].detach().cpu().numpy()
            idh_pred = torch.argmax(idh_p, dim=1).detach().cpu().numpy()
            grade_pred= torch.argmax(grade_p, dim=1).detach().cpu().numpy()
            idh_true = idh.detach().cpu().numpy()
            grade_true= grade.detach().cpu().numpy()
            idh_pred_scores.append(idh_pred_score)
            grade_pred_scores.append(grade_pred_score)
            idh_preds.append(idh_pred)
            grade_preds.append(grade_pred)
            idh_trues.append(idh_true)
            grade_trues.append(grade_true)
            lossidh = criterion1(idh_p, idh)
            lossgrade=criterion2(grade_p,grade)
            loss = nets['mtl'](segloss0, segloss1,segloss2,segloss3,lossidh, lossgrade)
            loss.backward()  # 反向传播，计算梯度
            train_loss.append(loss.item())
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 重置梯度

        print(
            f'Epoch {epoch + 1}/{args.end_epoch}, Average WT Dice: {sum(wt_dices) / len(wt_dices)}, Average TC Dice: {sum(tc_dices) / len(tc_dices)}, Average ET Dice: {sum(et_dices) / len(et_dices)}')

        print('总损失:', sum(train_loss) / len(train_loss))
        idh_trues = np.concatenate(idh_trues).tolist()
        idh_preds = np.concatenate(idh_preds).tolist()
        idh_pred_scores = np.concatenate(idh_pred_scores).tolist()

        grade_trues = np.concatenate(grade_trues).tolist()
        grade_preds = np.concatenate(grade_preds).tolist()
        grade_pred_scores = np.concatenate(grade_pred_scores).tolist()

        print('idh真实的结果')
        print(idh_trues)
        print('idh预测的结果')
        print(idh_preds)

        print('grade真实的结果')
        print(grade_trues)
        print('grade预测的结果')
        print(grade_preds)
        results_idh_train = evalution_metirc_boostrap(
            y_true=idh_trues,
            y_pred_score=idh_pred_scores,
            y_pred=idh_preds,
            labels=labels,
            target_names=idh_names
        )
        results_grade_train = evalution_metirc_boostrap(
            y_true=grade_trues,
            y_pred_score=grade_pred_scores,
            y_pred=grade_preds,
            labels=labels,
            target_names=grade_names
        )

        wt_dices_val, tc_dices_val, et_dices_val = [], [], []
        idh_pred_scores_val = []
        grade_pred_scores_val = []
        idh_preds_val = []
        grade_preds_val = []
        idh_trues_val = []
        grade_trues_val = []
        with torch.no_grad():
            nets['model'].eval()
            nets['t2f'].eval()
            nets['fusion'].eval()
            nets['idh'].eval()
            nets['grade'].eval()
            nets['mtl'].eval()
            for i, data in enumerate(val_loader):
                x, target, idh, grade, name = data
                x = x.cuda(args.local_rank, non_blocking=True).float()
                target = target.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)
                grade = grade.cuda(args.local_rank, non_blocking=True)
                feature_en, seg_output = nets['model'](x)
                pred_labels = torch.argmax(seg_output, dim=1)
                mapping = {3: 4}
                for k, v in mapping.items():
                    pred_labels[pred_labels == k] = v
                # 计算每个样本的 Dice 系数并保存
                for idx in range(pred_labels.shape[0]):
                    wt_dice = dice_coeff(pred_labels[idx], target[idx], [1, 2, 4])
                    tc_dice = dice_coeff(pred_labels[idx], target[idx], [1, 4])
                    et_dice = dice_coeff(pred_labels[idx], target[idx], [4])
                    wt_dices_val.append(wt_dice.item())
                    tc_dices_val.append(tc_dice.item())
                    et_dices_val.append(et_dice.item())
                flair_image = x[:, 0, :, :, :].unsqueeze(1)
                t2_image = x[:, 3, :, :, :].unsqueeze(1)
                featuret2f = nets['t2f'](t2_image, flair_image)
                output = nets['fusion'](feature_en, featuret2f)
                idh_p = nets['idh'](output)
                grade_p = nets['grade'](output)
                idh_pred_score = F.softmax(idh_p, dim=1)[:, 1].detach().cpu().numpy()
                grade_pred_score = F.softmax(grade_p, dim=1)[:, 1].detach().cpu().numpy()
                idh_pred = torch.argmax(idh_p, dim=1).detach().cpu().numpy()
                grade_pred = torch.argmax(grade_p, dim=1).detach().cpu().numpy()
                idh_true = idh.detach().cpu().numpy()
                grade_true = grade.detach().cpu().numpy()
                idh_pred_scores_val.append(idh_pred_score)
                grade_pred_scores_val.append(grade_pred_score)
                idh_preds_val.append(idh_pred)
                grade_preds_val.append(grade_pred)
                idh_trues_val.append(idh_true)
                grade_trues_val.append(grade_true)

            idh_trues_val = np.concatenate(idh_trues_val).tolist()
            idh_preds_val = np.concatenate(idh_preds_val).tolist()
            idh_pred_scores_val = np.concatenate(idh_pred_scores_val).tolist()

            grade_trues_val = np.concatenate(grade_trues_val).tolist()
            grade_preds_val = np.concatenate(grade_preds_val).tolist()
            grade_pred_scores_val = np.concatenate(grade_pred_scores_val).tolist()

            results_idh_val = evalution_metirc_boostrap(
                y_true=idh_trues_val,
                y_pred_score=idh_pred_scores_val,
                y_pred=idh_preds_val,
                labels=labels,
                target_names=idh_names
            )

            results_grade_val = evalution_metirc_boostrap(
                y_true=grade_trues_val,
                y_pred_score=grade_pred_scores_val,
                y_pred=grade_preds_val,
                labels=labels,
                target_names=grade_names
            )

            print('验证集')
            print(f'Epoch {epoch + 1}/{args.end_epoch}, Average WT Dice: {sum(wt_dices_val) / len(wt_dices_val)}, '
                  f'Average TC Dice: {sum(tc_dices_val) / len(tc_dices_val)}, '
                  f'Average ET Dice: {sum(et_dices_val) / len(et_dices_val)}')

            av_wt_dices_val = sum(wt_dices_val) / len(wt_dices_val)
            av_tc_dices_val = sum(tc_dices_val) / len(tc_dices_val)
            av_et_dices_val = sum(et_dices_val) / len(et_dices_val)
            av_dices_val = (av_wt_dices_val + av_tc_dices_val + av_et_dices_val) / 3

            # Save model if it's the best
            if av_dices_val > best_dice and results_idh_val['AUC'][0] > best_idh_auc and results_grade_val['AUC'][
                0] > best_grade_auc:
                best_dice = av_dices_val
                best_idh_auc = results_idh_val['AUC'][0]
                best_grade_auc = results_grade_val['AUC'][0]

                model_filename="best_modal.pth"
                # Save the model checkpoint
                model_path = os.path.join(checkpoint_dir, model_filename)
                torch.save({
                    'epoch': epoch,
                    'seg': nets['model'].state_dict(),
                    't2f': nets['t2f'].state_dict(),
                    'fusion': nets['fusion'].state_dict(),
                    'idh': nets['idh'].state_dict(),
                    'grade': nets['grade'].state_dict(),
                    'mtl': nets['mtl'].state_dict(),
                }, model_path)
                print(f"Model saved to {model_path}")

def dice_coeff(pred, target, class_labels):
        class_labels = torch.tensor(class_labels, device=pred.device)  # 将列表转换为张量
        pred = torch.isin(pred, class_labels).float()
        target = torch.isin(target, class_labels).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection+1e-6) / (pred.sum() + target.sum() + 1e-6)
        return dice

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)
def compute_dice_coefficient(pred, true,e=1e-6):
    """
    Computes the Dice coefficient, a measure of set similarity.
    """
    intersection = np.logical_and(pred, true).sum()
    union = pred.sum() + true.sum()
    dice = (2 * intersection+e) / (union+e)
    return dice
def Dice(output,target,weight=None, eps=1e-6):
    target = target.float()
    if weight is None:
        num = 2 * (output * target).sum()+eps
        den = output.sum() + target.sum() + eps
    else:

        num = 2 * (weight * output * target).sum()+eps
        den = (weight*output).sum() + (weight*target).sum() + eps
    return 1.0 - num/den

def expand_target(x, n_class, mode='softmax'):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape).to(x.device)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    return xx

def Dual_focal_loss(output, target):

    target[target == 4] = 3  # 转换标签 4 为 3
    target = expand_target(target, n_class=output.size()[1])  # 扩展目标标签为 one-hot 编码

    # 排除第 0 通道，不进行计算
    output = output[:, 1:, :, :, :]  # 保留通道 1, 2, 3
    target = target[:, 1:, :, :, :]  # 保留通道 1, 2, 3

    # 重新排列维度，方便后续计算
    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()

    # 展平张量以进行逐像素损失计算
    target = target.view(3, -1)  # 只保留 3 个通道
    output = output.view(3, -1)  # 只保留 3 个通道

    # 计算 focal 损失，忽略第 0 通道
    return -(F.log_softmax((1 - (target - output) ** 2), 0)).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def initialize_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def analyze_pred_labels(pred_labels):
    # 获取 pred_labels 中的唯一值
    unique_values, counts = torch.unique(pred_labels, return_counts=True)

    # 打印唯一值及其对应的个数
    for value, count in zip(unique_values.tolist(), counts.tolist()):
        print(f"Value: {value}, Count: {count}")

def evalution_metirc_boostrap(y_true, y_pred_score, y_pred, labels, target_names):
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)
    y_pred = np.array(y_pred)
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

    auc_ = roc_auc_score(y_true, y_pred_score)
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    accuracy_ = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity_ = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity_ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    F1_score_ = f1_score(y_true, y_pred, labels=labels, pos_label=1)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_AUC = []
    bootstrapped_ACC = []
    bootstrapped_SEN = []
    bootstrapped_SPE = []
    bootstrapped_F1 = []
    rng = np.random.RandomState(rng_seed)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_score), len(y_pred_score))
        if len(np.unique(y_true[indices.astype(int)])) < 2:
            # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
            continue
        auc = roc_auc_score(y_true[indices], y_pred_score[indices])
        bootstrapped_AUC.append(auc)

        confusion = confusion_matrix(y_true[indices], y_pred[indices])
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        F1_score = f1_score(y_true[indices], y_pred[indices], labels=labels, pos_label=1)

        bootstrapped_ACC.append(accuracy)
        bootstrapped_SPE.append(specificity)
        bootstrapped_SEN.append(sensitivity)
        bootstrapped_F1.append(F1_score)

    sorted_AUC = np.array(bootstrapped_AUC)
    sorted_AUC.sort()
    sorted_ACC = np.array(bootstrapped_ACC)
    sorted_ACC.sort()
    sorted_SPE = np.array(bootstrapped_SPE)
    sorted_SPE.sort()
    sorted_SEN = np.array(bootstrapped_SEN)
    sorted_SEN.sort()
    sorted_F1 = np.array(bootstrapped_F1)
    sorted_F1.sort()

    results = {
        'AUC': (auc_, sorted_AUC[int(0.05 * len(sorted_AUC))], sorted_AUC[int(0.95 * len(sorted_AUC))]),
        'Accuracy': (accuracy_, sorted_ACC[int(0.05 * len(sorted_ACC))], sorted_ACC[int(0.95 * len(sorted_ACC))]),
        'Specificity': (specificity_, sorted_SPE[int(0.05 * len(sorted_SPE))], sorted_SPE[int(0.95 * len(sorted_SPE))]),
        'Sensitivity': (sensitivity_, sorted_SEN[int(0.05 * len(sorted_SEN))], sorted_SEN[int(0.95 * len(sorted_SEN))]),
        'F1_score': (F1_score_, sorted_F1[int(0.05 * len(sorted_F1))], sorted_F1[int(0.95 * len(sorted_F1))])
    }

    print("Confidence interval for the AUC: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['AUC']))
    print("Confidence interval for the Accuracy: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Accuracy']))
    print("Confidence interval for the Specificity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Specificity']))
    print("Confidence interval for the Sensitivity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Sensitivity']))
    print("Confidence interval for the F1_score: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['F1_score']))

    return results



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()


