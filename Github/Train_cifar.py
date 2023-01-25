from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--start_fix', default=10, type=int)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--threshold', default=0.9, type=int)
parser.add_argument('--data_path', default='/data/yirumeng/Dataset/cifar-10-batches-py', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--out', default='/', type=str)
args = parser.parse_args()

# torch.cuda.set_device(args.gpuid)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def ova_warmup(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    new_label_sp_neg = torch.zeros_like(label_sp_neg)
    a = [t for t in range(args.num_class)]
    for i in range(label.size(0)):
        choice = int(np.random.choice(a, 1))
        while choice == label[i]:
            choice = int(np.random.choice(a, 1))
        new_label_sp_neg[i, choice] = 1
    open_loss_neg = torch.mean(torch.sum(-torch.log(logits_open[:, 0, :] + 1e-8) * new_label_sp_neg, 1))

    Lo = open_loss_neg + open_loss
    return Lo

def ova_all(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.sum(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1))
    Lo = open_loss_neg + open_loss
    return Lo

def ova_loss(logits, logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    new_label_sp_neg = torch.zeros_like(label_sp_neg)
    prob_sorted_index = torch.argsort(logits, descending=True)
    for i in range(logits.size(0)):
        if prob_sorted_index[i, 0] == label[i]:
            new_label_sp_neg[i, prob_sorted_index[i, 1]] = 1
            new_label_sp_neg[i, np.random.choice(prob_sorted_index[i, 2:].cpu(), 1)] = 1
        else:
            new_label_sp_neg[i, prob_sorted_index[i, 0]] = 1
            choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
            while choice == label[i]:
                choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
            new_label_sp_neg[i, choice] = 1
    open_loss_neg = torch.mean(torch.sum(-torch.log(logits_open[:, 0, :] + 1e-8) * new_label_sp_neg, 1))

    Lo = open_loss_neg + open_loss
    return Lo



def save_checkpoint(state, checkpoint, epoch, filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, str(epoch) + "_" + filename)
    torch.save(state, filepath)


# Training
def train(epoch, net, rotnet_head,optimizer, labeled_trainloader, unlabeled_trainloader, ova_trainloader, unova_trainloader):
    net.train()
    rotnet_head.train()
    unlabeled_train_iter = iter(unlabeled_trainloader)
    labeled_train_iter = iter(labeled_trainloader)
    ova_train_iter = iter(ova_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx in range(900):
        try:
            inputs_x,inputs_x1,labels_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x,inputs_x1,labels_x = labeled_train_iter.next()
        try:
            inputs_o,inputs_o1,labels_o = ova_train_iter.next()
        except:
            ova_train_iter = iter(ova_trainloader)
            inputs_o,inputs_o1,labels_o = ova_train_iter.next()
        try:
            inputs_u,inputs_u1,labels_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u,inputs_u1,labels_u = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        tmp_labels_o = labels_o
        tmp_labels_o = tmp_labels_o.cuda()
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        labels_o = torch.zeros(batch_size, args.num_class).scatter_(1, labels_o.view(-1, 1), 1)
        inputs_x,inputs_x1,labels_x = inputs_x.cuda(),inputs_x1.cuda(),labels_x.cuda()
        inputs_o, inputs_o1,labels_o = inputs_o.cuda(),inputs_o1.cuda(), labels_o.cuda()
        inputs_u, inputs_u1, labels_u = inputs_u.cuda(),inputs_u1.cuda(),labels_u.cuda()
        with torch.no_grad():
            ##Label guessing
            outputs_u, outputs_open_u, _ = net(inputs_u)
            outputs_u1, outputs_open_u1, _ = net(inputs_u1)
            pu = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u1, dim=1)) / 2
            ptu = pu ** (1 / args.T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            ##Label refinement
            outputs_x, outputs_open_x, _ = net(inputs_x)
            outputs_x1, outputs_open_x1, _ = net(inputs_x1)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x1, dim=1)) / 2
            out_open1 = F.softmax(outputs_open_x.view(outputs_open_x.size(0), 2, -1), 1)
            out_open1 = out_open1[:, 1, :]
            w_x1, _ = torch.max(out_open1, 1)
            out_open2 = F.softmax(outputs_open_x1.view(outputs_open_x1.size(0), 2, -1), 1)
            out_open2 = out_open2[:, 1, :]
            w_x2, _ = torch.max(out_open2, 1)
            w_x = (w_x1 + w_x2) / 2
            w_x = w_x.view(-1, 1).type(torch.FloatTensor).cuda()
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x, inputs_x1, inputs_u, inputs_u1], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits,logits_open, _ = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)


        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        outputs_o, outputs_open_o, _ = net(inputs_o)
        if args.r == 0.8 or args.r == 0.9:
            Lo = ova_warmup(outputs_open_o, tmp_labels_o)
        else:
            Lo = ova_loss(outputs_o, outputs_open_o, tmp_labels_o)
        # rotate unlabeled data with 0, 90, 180, 270 degrees
        inputs_r_x = torch.cat(
            [torch.rot90(inputs_x, i, [2, 3]) for i in range(4)], dim=0)
        targets_r_x = torch.cat(
            [torch.empty(labels_x.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
        inputs_r_x = inputs_r_x.cuda()
        _, _, feats_r_x = net(inputs_r_x)
        Lr_x = F.cross_entropy(rotnet_head(feats_r_x), targets_r_x, reduction='mean')
        inputs_r_u = torch.cat(
            [torch.rot90(inputs_u, i, [2, 3]) for i in range(4)], dim=0)
        targets_r_u = torch.cat(
            [torch.empty(labels_u.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
        inputs_r_u = inputs_r_u.cuda()
        _, _, feats_r_u = net(inputs_r_u)
        Lr_u = F.cross_entropy(rotnet_head(feats_r_u), targets_r_u, reduction='mean')
        loss = Lx + lamb*Lu + penalty +Lo +Lr_u+Lr_x
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item()))
        sys.stdout.flush()


def warmup(epoch, net,rotnet_head, optimizer, dataloader):
    net.train()
    rotnet_head.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs,_, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs, open_outputs, _ = net(inputs)
        loss = CEloss(outputs, labels)
        loss_ova = ova_warmup(open_outputs, labels)
        # rotate unlabeled data with 0, 90, 180, 270 degrees
        inputs_r = torch.cat(
            [torch.rot90(inputs, i, [2, 3]) for i in range(4)], dim=0)
        targets_r = torch.cat(
            [torch.empty(labels.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
        inputs_r = inputs_r.cuda()
        _, _, feats_r = net(inputs_r)
        Lr = F.cross_entropy(rotnet_head(feats_r), targets_r, reduction='mean')
        if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty + loss_ova + Lr
        elif args.noise_mode == 'sym':
            L = loss + loss_ova + Lr
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


def test(epoch, net1):
    net1.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, open_output1, _ = net1(inputs)
            outputs1 = F.softmax(outputs1, 1)
            open_output1 = F.softmax(open_output1.view(open_output1.size(0), 2, -1), 1)
            open_output1 = F.softmax(open_output1[:, 1, :], 1)
            outputs = outputs1 + open_output1
            _, predicted1 = torch.max(outputs, 1)
            _, predicted2 = torch.max(outputs1, 1)
            _, predicted3 = torch.max(open_output1, 1)

            total += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()
            correct2 += predicted2.eq(targets).cpu().sum().item()
            correct3 += predicted3.eq(targets).cpu().sum().item()
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    acc3 = 100. * correct3 / total
    print("\n| Test Epoch #%d\t Accuracy_all: %.2f%%\t Accuracy_fc: %.2f%%\t Accuracy_ova: %.2f%%\n" % (
    epoch, acc1, acc2, acc3))
    test_log.write(
        'Epoch:%d   Accuracy_all: %.2f%%\t Accuracy_fc: %.2f%%\t Accuracy_ova: %.2f%%\n' % (epoch, acc1, acc2, acc3))
    test_log.flush()

def eval_train(model, use_ova, noise_or_not,epoch):
    model.eval()
    clean_idx = []
    targets_list = []
    ova_list = []
    beta_class = [0] * args.num_class
    loss_list = []
    target_list = []
    index_list = []
    all_score_w = [0] * len(eval_loader.dataset)
    all_noise_not = [0] * len(eval_loader.dataset)

    with torch.no_grad():
        for batch_idx, (inputs,_, targets, index) in enumerate(eval_loader):
            inputs = inputs.cuda()
            outputs, outputs_open, _ = model(inputs)
            logits = F.softmax(outputs, 1)
            max_score, pred_label = torch.max(logits, dim=1)
            max_score = max_score.cpu().numpy().tolist()
            pred_label = pred_label.cpu().numpy().tolist()
            for score, label in zip(max_score, pred_label):
                if score > args.threshold:
                    beta_class[label] += 1
        class_beta_score = {}
        max_t = np.max(np.array(beta_class))
        for i in range(args.num_class):
            class_beta_score[i] = beta_class[i] / max_t
    with torch.no_grad():
        for batch_idx, (inputs,inputs1, targets, index) in enumerate(eval_loader):
            tmp_targets = targets
            inputs, inputs1, targets = inputs.cuda(), inputs1.cuda(), targets.cuda()
            inputs_all = torch.cat([inputs, inputs1], dim=0)
            outputs_all, outputs_open_all, _ = model(inputs_all)
            outputs, outputs1 = outputs_all.chunk(2)
            _, pred_label = torch.max(outputs, dim=1)
            pred_same = pred_label == targets
            pred_same = pred_same.cpu().tolist()
            confidence = F.softmax(outputs, dim=1)
            label_range = torch.arange(0, outputs.size(0)).long()
            target_confidence = confidence[label_range, targets.cpu()]
            target_confidence = target_confidence.cpu().numpy().tolist()
            pred_label = pred_label.cpu().numpy().tolist()
            for i, (score, p_label, is_same) in enumerate(zip(target_confidence, pred_label, pred_same)):
                if not is_same:
                    continue
                if use_ova:
                    if score > args.threshold * class_beta_score[p_label]:
                        ova_list.append(index[i].item())
                        targets_list.append(tmp_targets[i].item())
            outputs_open, outputs_open1 = outputs_open_all.chunk(2)
            out_open1 = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_outputs_open = out_open1
            out_open1 = out_open1[:,1,:]
            _, predicted1 = torch.max(out_open1, 1)
            pred_ova_same1 = predicted1 == targets
            pred_ova_same1 = pred_ova_same1.cpu().tolist()
            out_open2 = F.softmax(outputs_open1.view(outputs_open1.size(0), 2, -1), 1)
            out_open2 = out_open2[:,1,:]
            _, predicted2 = torch.max(out_open2, 1)
            pred_ova_same2 = predicted2 == targets
            pred_ova_same2 = pred_ova_same2.cpu().tolist()
            tmp_range = torch.arange(0, tmp_outputs_open.size(0)).long().cuda()
            score_w = tmp_outputs_open[tmp_range, 1, targets]
            score_w = score_w.cpu().numpy().tolist()
            for single_index, single_score_w in zip(index, score_w):
                all_score_w[single_index] = single_score_w
                if noise_or_not[single_index]:
                    all_noise_not[single_index] = 1
            if use_ova:
                for i, (p1, p2) in enumerate(zip(pred_ova_same1, pred_ova_same2)):
                    if p1==1:
                    # if p1==1 and p2 ==1:
                        clean_idx.append(index[i].item())
    with torch.no_grad():
        for batch_idx, (inputs,_, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduce=False, reduction='none')
            loss = loss.cpu().numpy().tolist()
            loss_list.extend(loss)
            index_list.extend(index.numpy().tolist())
            target_list.extend(targets.cpu().numpy().tolist())

    loss_dict = {}
    index_dict = {}
    for single_loss,target,index in zip(loss_list,target_list,index_list):
        if target not in loss_dict:
            loss_dict[target] = []
            index_dict[target] = []
        loss_dict[target].append(single_loss)
        index_dict[target].append(index)

    for target in loss_dict:
        loss_list = np.array(loss_dict[target])
        ind_sorted = np.argsort(loss_list)
        num_remember = int(0.01 * len(loss_list))
        ind_update = ind_sorted[:num_remember]
        for i in ind_update:
            clean_idx.append(index_dict[target][i])
            ova_list.append(index_dict[target][i])
    from sklearn.metrics import roc_auc_score
    all_score_w = np.array(all_score_w)
    all_noise_not = np.array(all_noise_not)
    auc = roc_auc_score(all_noise_not, all_score_w)
    clean_idx = list(set(clean_idx))
    ova_list = list(set(ova_list))
    pure1 = np.sum(noise_or_not[ova_list])
    pure2 = np.sum(noise_or_not[clean_idx])
    pure_log.write(
        'Epoch:%d   Pure_ws:%d/%d    %.4f    Pure_ova:%d/%d    %.4f   auc:%.4f\n' % (
        epoch, pure1, len(ova_list), pure1 / len(ova_list), pure2, len(clean_idx), pure2 / len(clean_idx), auc))
    pure_log.flush()
    return clean_idx, ova_list


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    rotnet_head = torch.nn.Linear(512, 4)
    rotnet_head = rotnet_head.cuda()
    return model,rotnet_head



stats_log = open(os.path.join(args.out,'%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt') , 'w')
test_log = open(os.path.join(args.out,'%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt'), 'w')
pure_log = open(os.path.join(args.out,'%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_pure.txt'), 'w')

if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                     num_workers=5, \
                                     root_dir=args.data_path, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1,rotnet_head = create_model()
cudnn.benchmark = True
grouped_parameters = [
    {'params': net1.parameters()},
    {'params': rotnet_head.parameters()}
]
criterion = SemiLoss()
optimizer1 = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)


class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(1)
        return loss


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()
CE_soft = SoftEntropy()
all_loss = [[], []]  # save the history of losses from two networks

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 100
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    warmup_trainloader, noise_or_not = loader.run('warmup')
    if epoch < warm_up:
        print('Warmup Net1')
        warmup(epoch, net1, rotnet_head,optimizer1, warmup_trainloader)
    else:
        clean_idx, ova_list = eval_train(net1, epoch > 9, noise_or_not, epoch)
        labeled_trainloader, unlabeled_trainloader, ova_trainloader, unova_trainloader = loader.run('train', clean_idx, ova_list)
        print('Train Net1')
        train(epoch, net1, rotnet_head,optimizer1, labeled_trainloader, unlabeled_trainloader, ova_trainloader, unova_trainloader)    # train net1

    test(epoch, net1)
