import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ContrastiveLoss import *
from Config import *
from SiameseNetworkDataset import *
from SiameseNetwork import *

def imshow(img,text=None,should_save=False):
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


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((128, 128)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)
testData = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)


# list = [0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,
# 0.575,0.600,0.625,0.650,0.675,0.700,0.725,0.750,0.775,0.800,
# 0.820,0.840,0.860,0.880,0.900,0.920,0.940,0.950,0.960,0.970,0.980,0.990]

list = [0.825]
c = ['{:.3f}'.format(i) for i in list]
for cnt,loop in enumerate(c):
    newnet = SiameseNetwork().cuda()
    newnet.eval()
    newnet.load_state_dict(torch.load('./checkpoints/0.765loop9.pkl'))
    num_of_gap = 200
    start_distance = 0.0
    end_distance = 2.0
    axis_x = np.linspace(start_distance, end_distance, num_of_gap+1)
    axis_y_same = [0 for _ in range(num_of_gap+1)]
    axis_y_diff = [0 for _ in range(num_of_gap+1)]
    count_same = 0
    count_diff = 0
    for epoch in range(0,Config.test_number_epochs):
        for i, data in enumerate(testData,0):
            print(epoch)
            img0, img1 , label,img0_tuple,img1_tuple = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1,output2 = newnet(img0,img1)
            a = torch.zeros(output1.size(dim=1), output1.size(dim=0))
            a = a.cuda()
            b = torch.zeros(output2.size(dim=1), output2.size(dim=0))
            b = b.cuda()
            output1 = torch.mul(output1, ((1 / torch.norm(output1, dim=1)).transpose(0, -1) + a).t())
            output2 = torch.mul(output2, ((1 / torch.norm(output2, dim=1)).transpose(0, -1) + b).t())
            euclidean_distance= F.pairwise_distance(output1, output2, keepdim = True)
            print(label.item())
            print(euclidean_distance.item())
            euclidean_distance_write = euclidean_distance.item()
            img0_path_write = img0_tuple[0][0]
            img1_path_write = img1_tuple[0][0]
            # f = open('EER_imagePairs_Address_Gabor.txt', 'a')
            # f.write(str(euclidean_distance_write))
            # f.write(' '+img0_path_write)
            # f.write(' '+img1_path_write)
            # f.write(' '+loop)
            # f.write('\n')
            temp = int(euclidean_distance / ((end_distance-start_distance)/num_of_gap))
            if label.item() == 0:
                count_same = count_same + 1
                if temp<num_of_gap:
                    axis_y_same[temp] +=1
                else:
                    axis_y_same[num_of_gap] +=1
            if label.item() == 1:
                count_diff = count_diff + 1
                if temp<num_of_gap:
                    axis_y_diff[temp] +=1
                else:
                    axis_y_diff[num_of_gap] +=1

    for item,x in enumerate(axis_y_same):
        axis_y_same[item] = x/count_same
    for item,x in enumerate(axis_y_diff):
        axis_y_diff[item] = x/count_diff
    axis_y_same[0] = 0

    # plt.plot(axis_x, axis_y_same, 'g', axis_x, axis_y_diff, 'r')
    # plt.show()

    figsize = 11, 9
    figure, ax = plt.subplots(figsize=figsize)

    # 在同一幅图片上画两条折线
    A, = plt.plot(axis_x, axis_y_same, 'r', label='Matched Patch', linewidth=1.0)
    B, = plt.plot(axis_x, axis_y_diff, 'b', label='Non-matched Patch', linewidth=1.0)

    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    legend = plt.legend(handles=[A, B], prop=font1)

    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=13)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 19,
             }
    plt.xlabel('Euclidean Distance', font2)
    plt.ylabel('Percentage', font2)
    plt.title('')
    # 将文件保存至文件中并且画出图
    plt.savefig('SGD' + loop + 'gabor' + '.eps')
    plt.clf()  #remember to clear the figure, or the latter figure will overlap the former figure since plt is used for many times in the loop
    # plt.show()

    f = open('EER_record.txt','a')
    for i in range(1,num_of_gap+1):
        FRR = 0
        FAR = 0
        for m in range(i,num_of_gap+1):
            FRR = FRR+axis_y_same[m]
        for n in range(0,i):
            FAR = FAR + axis_y_diff[n]
        EER = 0.5*(FRR + FAR)
        print('Distance:',i*((end_distance-start_distance)/num_of_gap),'EER:',EER,'LOOP:',loop)
        f.write('Distance: ')
        f.write(str(i * ((end_distance-start_distance)/num_of_gap)))
        f.write('    EER: ')
        f.write(str(EER))
        f.write('    LOOP: ')
        f.write(loop)
        f.write('\n')

    print('diff',count_diff,'loop',loop)
    print('same',count_same,'loop',loop)
    f.write('count_diff:')
    f.write(str(count_diff))
    f.write('    count_same:')
    f.write(str(count_same))
    f.write('\n')
    f.close()