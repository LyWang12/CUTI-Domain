import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.getdata import Cus_Dataset
from utils.util import get_home_data

from utils.network import ImageClassifier
from transformer import swin_tiny_patch4_window7_224

SEED = 2022
lr = 0.0001
nepoch = 30
workers = 10
batch_size = 32
device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SOURCE = 'Art'   # Art, Clipart, Product, RealWorld
TARGET = 'Clipart'   # Art, Clipart, Product, RealWorld

def validate_class(val_loader, model, epoch, num_class=10):
    model.eval()

    correct = 0
    total = 0
    c_class = [0 for i in range(num_class)]
    t_class = [0 for i in range(num_class)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels[:, 0], axis=1)
        correct += (predicted == true_label).sum().item()
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1

    acc = 100.0 * correct / total
    print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=acc))
    model.train()
    return acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    num_classes = 65

    dataset1 = get_home_data(SOURCE)
    dataset2 = get_home_data(TARGET)
    print('original data loaded...')
    PATH = 'D:/WLY/Documents/NUAA/CVPR/github/' + SOURCE + '_to_' + TARGET + '.pth'

    datafile = Cus_Dataset(mode='train',
                           dataset_1=dataset1, begin_ind1=0, size1=2000,
                           dataset_2=dataset2, begin_ind2=0, size2=2000)
    datafile_val1 = Cus_Dataset(mode='val', dataset_1=dataset1, begin_ind1=2000, size1=400)
    datafile_val2 = Cus_Dataset(mode='val', dataset_1=dataset2, begin_ind1=2000, size1=400)
    valloader1 = DataLoader(datafile_val1, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader2 = DataLoader(datafile_val2, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    backbone = swin_tiny_patch4_window7_224()
    weights_dict = torch.load('swin_tiny_patch4_window7_224.pth')["model"]
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    print(backbone.load_state_dict(weights_dict, strict=False))
    model = ImageClassifier(backbone, num_classes).cuda()
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch: 0.999 ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()

    for epoch in range(nepoch):
        for i, (img1, label1, img2, label2) in enumerate(dataloader):
            img1, label1, img2, label2 = img1.to(device), label1.to(device), img2.to(device), label2.to(device)

            img1.float()
            label1 = label1.float()
            img2.float()
            label2 = label2.float()

            out1, out2 = model(x1=img1, x2=img2, choice=i)
            out1 = F.log_softmax(out1, dim=1)
            loss1 = criterion_KL(out1, label1)

            out2 = F.log_softmax(out2, dim=1)
            loss2 = criterion_KL(out2, label2)  # ?change to 0.01 when different dataset, 0.1 on watermark
            alpha = 0.1
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            loss = loss1 - loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # test
        acc1 = validate_class(valloader1, model, epoch, num_class=num_classes)
        acc2 = validate_class(valloader2, model, epoch, num_class=num_classes)
        f = open(PATH.split('.')[0] + '.txt', "a+")
        f.write("epoch = {:02d}, acc_s = {:.3f}, acc_t = {:.3f}".format(epoch, acc1, acc2) + '\n')
        f.close()
    torch.save(model.state_dict(), PATH)

            


if __name__ == "__main__":
    setup_seed(SEED)
    train()
