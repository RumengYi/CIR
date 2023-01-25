from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from randaugment import RandAugmentMC
from numpy.testing import assert_array_almost_equal

# from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict
def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', clean_idx=[], log='',
                 transform_s=None,transform_w2=None,transform_s2=None):

        self.r = r  # noise ratio
        self.transform = transform
        self.transform_w2 = transform_w2
        self.transform_s = transform_s
        self.transform_s2 = transform_s2
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                if dataset == 'cifar100' and noise_mode == 'asym':
                    noise_label = noisify_cifar100_asymmetric(np.array(train_label), self.r, random_state=0).tolist()
                # print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))
            self.noise_or_not = np.transpose(noise_label) == np.transpose(train_label)
            # print("noise or not sum is %d"%(np.sum(self.noise_or_not)))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = clean_idx

                elif self.mode == "unlabeled":
                    total_index = [i for i in range(50000)]
                    pred_idx = [x for x in total_index if x not in clean_idx]

                self.train_data = train_data[pred_idx, ...]
                self.noise_label = [noise_label[i] for i in pred_idx]

                noise_or_not = np.array(self.noise_or_not)
                print("noise or not sum is %d" % (np.sum(noise_or_not)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_w2(img)
            return img1, img2,target
        elif self.mode == 'unlabeled':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_w2(img)
            return img1, img2,target
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
            self.transform_train_s = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            self.transform_train_s = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, clean_idx=[], ova_list=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file, transform_s=self.transform_train_s)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)

            return trainloader, all_dataset.noise_or_not

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, transform_w2=self.transform_train,transform_s=self.transform_train_s,transform_s2=self.transform_train_s, mode="labeled",
                                            noise_file=self.noise_file, clean_idx=clean_idx, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            ova_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, transform_w2=self.transform_train,transform_s=self.transform_train_s,transform_s2=self.transform_train_s, mode="labeled",
                                        noise_file=self.noise_file, clean_idx=ova_list, log=self.log)
            ova_trainloader = DataLoader(
                dataset=ova_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, transform_w2=self.transform_train,transform_s=self.transform_train_s,transform_s2=self.transform_train_s, mode="unlabeled",
                                              noise_file=self.noise_file, clean_idx=clean_idx)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            unova_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, transform_w2=self.transform_train,transform_s=self.transform_train_s,transform_s2=self.transform_train_s, mode="unlabeled",
                                        noise_file=self.noise_file, clean_idx=ova_list, log=self.log)
            unova_trainloader = DataLoader(
                dataset=unova_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)

            return labeled_trainloader, unlabeled_trainloader, ova_trainloader, unova_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_train, mode='all',
                                         noise_file=self.noise_file, transform_s=self.transform_train_s)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
