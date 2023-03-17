import torch
import struct


def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('resnet50.pth')
    # net = net.to('cuda:0')
    net.eval()
    print('model: ', net)
    tmp = torch.ones(1, 3, 224, 224)
    print('input: ', tmp)
    out = net(tmp)
    print('output:', out)

    f = open("resnet50.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()

