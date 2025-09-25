import torch.optim as optim
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix, fbeta_score, \
    precision_score, recall_score
import torch
import numpy as np
import math
import copy
import torch.nn.functional as F
import sys


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay', default=50)
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio', default=0.8)
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.', default=0)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def CE_loss(pred, label):
    label = label.type(torch.int64)
    loss = nn.CrossEntropyLoss()
    return loss(pred, label)

def BCE_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1).type(torch.float32)
    loss = nn.BCELoss()
    return loss(pred, label)

def get_acc_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return acc_score

def get_f1_scores(pred, label):
    # Detach the prediction and move to CPU if necessary
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)  # Get the class index with the highest probability

    # Convert to numpy arrays for use with sklearn metrics
    pred_label = pred_label.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    # Calculate F1 score for binary or multiclass classification
    f1 = f1_score(label, pred_label, average='micro')  # You can change the average method
    return f1

def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        #print(torch.unique(label).size(0))
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        print("he")
        return None
        pass


def get_scores(pred, label):
    auc_score = get_auc_score(pred, label)
    acc_score = get_acc_score(pred, label)
    

    return acc_score, auc_score

def get_micro_scores(pred, label):
    acc_score = get_acc_score(pred, label)
    f1_score = get_f1_scores(pred, label)
    

    return acc_score, f1_score

