import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import struct

# alexnet
def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.alexnet(pretrained=True)
    #net.fc = nn.Linear(512, 2)
    net.eval()
    net = net.to('cuda:0')
    print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('alexnet out:', out.shape)
    torch.save(net, "alexnet.pth")

    # 生成wts权重文件
    f = open("alexnet.wts", 'w')
    print('==net.state_dict().keys():', net.state_dict().keys())
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    print('==f:', f)

if __name__ == '__main__':
    main()