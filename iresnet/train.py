import os
import random
import sys
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

# write here or command is both fine
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5" #"0,1" if you want to use both
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 1 means make every thing synchronized, only use for debuging

import torch
from torch.nn import functional as F

from Net import Net
from dataset import NetData


def spectral_norm(module, coeff, iters):
    W = module.weight.data
    h, w = W.size()
    u = normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=1e-12)
    v = normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=1e-12)

    with torch.no_grad():
        for _ in range(iters):
            v = normalize(torch.mv(W.t(), u), dim=0, eps=1e-12, out=v)
            u = normalize(torch.mv(W, v), dim=0, eps=1e-12, out=u)

    sigma = torch.dot(u, torch.mv(W, v))
    factor = max(1.0, (sigma / coeff).item())
    weight = W / factor
    module.weight.data = weight


def log_loss(inputs, net):
    input_x = inputs
    process_x = inputs
    log_det_J = 0
    for i in range(net.block_num):
        process_x_act, J_act = net.block_list[i].actnorm(process_x)

        elu_grad = (process_x_act > 0.0) * 1.0
        elu_grad += (process_x_act <= 0.0) * (F.elu(process_x_act) + 1)
        elu_grad = torch.diag_embed(elu_grad)
        process_x = net.block_list[i].basic_block[1](net.block_list[i].basic_block[0](process_x_act))

        elu_grad_1 = (process_x > 0.0) * 1.0
        elu_grad_1 += (process_x <= 0.0) * (F.elu(process_x) + 1)
        elu_grad_1 = torch.diag_embed(elu_grad_1)
        process_x = net.block_list[i].basic_block[3](net.block_list[i].basic_block[2](process_x))


        elu_grad_2 = (process_x > 0.0) * 1.0
        elu_grad_2 += (process_x <= 0.0) * (F.elu(process_x) + 1)
        elu_grad_2 = torch.diag_embed(elu_grad_2)
        input_x = net.block_list[i].basic_block[5](net.block_list[i].basic_block[4](process_x)) + process_x_act
        process_x = input_x

        J_part2 = net.block_list[i].basic_block[5].weight @ elu_grad_2 @ net.block_list[i].basic_block[3].weight @ elu_grad_1 @ net.block_list[i].basic_block[1].weight @ elu_grad
        I = torch.eye(input_x.size(1)).expand_as(J_part2).cuda()
        J = I + J_part2
        log_det_J += torch.log(abs(torch.det(J))) + J_act

    log_all = log_det_J - torch.diag((input_x @ input_x.T) / 2)
    return -log_all.sum() / log_all.size(0)



def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    torch.manual_seed(1)
    print('begin initial...')
    lr = 0.000004
    resume_checkpoint_path = 'model.pt'
    # net = Net().cuda()
    net = torch.load(resume_checkpoint_path).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net_data = NetData()
    train_loader = torch.utils.data.DataLoader(net_data, batch_size=16, shuffle=True)
    epochs = 100
    all_loss = []
    # with torch.no_grad():
    #    net(net_data.get_all_items().cuda())
    print('finish initial...')
    print('begin train...')
    net.train()
    for epoch in range(epochs):
        if epoch%1 == 0:
            lr = lr * 0.8
        update_lr(optimizer, lr)
        each_epoch_loss= 0
        for batch_idx, data in enumerate(train_loader):
            # train loop
            data = data.cuda()
            loss = log_loss(data, net)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            for i in range(net.block_num):
                spectral_norm(net.block_list[i].basic_block[1], .87, 100)
                spectral_norm(net.block_list[i].basic_block[3], .87, 100)
                spectral_norm(net.block_list[i].basic_block[5], .87, 100)
            each_epoch_loss = each_epoch_loss + loss.item()
        print('Train Epoch:', epoch, 'Loss:', each_epoch_loss/len(train_loader))
        if len(all_loss)!= 0 and min(all_loss) > loss.item():
            torch.save(net, 'model.pt')
        all_loss.append(loss.item())
        inverse_input = torch.randn(200, 256).cuda()
        inverse_output = net.inverse(inverse_input)
        inverse_output = net_data.find_original(inverse_output)
        path = '../engn8536/Datasets/z_map/newc_' + str(epoch) + '.pt'
        torch.save(inverse_output, path)

    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.plot(range(len(all_loss)), all_loss, color='red', linewidth=2.0, linestyle='-')
    plt.savefig('plot.png', format='png')
