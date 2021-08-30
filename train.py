import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils
from torch import optim
from Config import *
from ContrastiveLoss import *
from SiameseNetwork import *
from SiameseNetworkDataset import *
import torch


def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((128,128)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

print(len(train_dataloader))

for loop in np.arange(0.765,0.915,0.030):
    counter = []
    iteration_number = 0
    criterion = ContrastiveLoss(loop)
    net = SiameseNetwork().cuda()
    net.train(mode=True)
    lr1 = 0.001  # 1
    momentum = 0.0001
    optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr1, momentum=momentum, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.95, patience=250,
                                                            min_lr=0.00008)
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label,img0_tuple,img1_tuple = data
            img0, img1 , label = img0.cuda(), img1.cuda(), label.cuda()
            # print(img0.size())
            optimizer1.zero_grad()
            output1,output2 = net(img0,img1)
            a = torch.zeros(output1.size(dim=1), output1.size(dim=0))
            a = a.cuda()
            b = torch.zeros(output2.size(dim=1), output2.size(dim=0))
            b = b.cuda()
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer1.step()
            scheduler1.step(loss_contrastive)
            if i%10==0:
                # print("Epoch number {} Loop number{} \n Current loss {}\n".format(epoch,loop,loss_contrastive.item()))
                print('batch: {}, loss: {}, lr: {}, loopNumber: {}, i: {}'.format(epoch, loss_contrastive, optimizer1.param_groups[0]['lr'],loop,i))
                iteration_number +=10
                counter.append(iteration_number)
        if epoch%5 == 4:
            torch.save(net.state_dict(),'./checkpoints'+'/'+str(loop)+'loop'+str(epoch)+'.pkl')
