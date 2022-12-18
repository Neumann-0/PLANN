# 筛选出图片中的重要区域

import copy
import numpy as np
import time
from PLNN import *
from changeSeq import *
from shapeletCandidates import *
import matplotlib.pyplot as plt
from PIL import Image

# 用于定位滤网的位置
def locate(x,y):
    result = [x*36+y,x*36+y+1,x*36+y+2,(x+1)*36+y,(x+1)*36+y+1,(x+1)*36+y+2,
              (x+2)*36+y,(x+2)*36+y+1,(x+2)*36+y+2]
    return result


# 添加扰动
def changeTSdata(TSdata,x,y):
    change_list = locate(x, y)
    seq_1 = incrementSeq(TSdata, change_list[0], change_list[2], TSdata.size)
    seq_2 = incrementSeq(TSdata, change_list[3], change_list[5], TSdata.size)
    seq_3 = incrementSeq(TSdata, change_list[6], change_list[8], TSdata.size)
    # seq_4 = incrementSeq(TSdata, change_list[12], change_list[15], TSdata.size)
    new_ts = TSdata + seq_1 + seq_2 + seq_3
    return new_ts


def calculate_inequality_cofficients_mnist(stage,H1,H2,H3, model, data, filename):
    states, output = model(data,stage)
    #    _, prediction = torch.max(output.data, 1)
    #   prediction = np.array(prediction)
    #  prediction = prediction.reshape(prediction.shape[0],1)
    #  print("prediction is ", prediction)
    w1, b1 = model.state_dict()['fc1.weight'], model.state_dict()['fc1.bias']
    w2, b2 = model.state_dict()['fc2.weight'], model.state_dict()['fc2.bias']
    w3, b3 = model.state_dict()['fc3.weight'], model.state_dict()['fc3.bias']
    w4, b4 = model.state_dict()['fc4.weight'], model.state_dict()['fc4.bias']

    diag_s1 = torch.diag(torch.tensor((states['h1']),
                                      dtype=torch.float32))
    w2_hat = torch.matmul(w2, torch.matmul(diag_s1, w1))
    b2_hat = torch.matmul(w2, torch.matmul(diag_s1, b1)) + b2

    diag_s2 = torch.diag(torch.tensor((states['h2']),
                                      dtype=torch.float32))

    w3_hat = torch.matmul(w3, torch.matmul(diag_s2, w2_hat))
    b3_hat = torch.matmul(w3, torch.matmul(diag_s2, b2_hat)) + b3
    #    print(w3_hat.size(), b3_hat.size())


    weights = torch.cat((w1, w2_hat, w3_hat)).numpy()
    bias = torch.cat((b1, b2_hat, b3_hat)).numpy()

    # bias = bias.reshape(22, 1)
    bias = bias.reshape(H1+H2+H3, 1)
    active_states = np.hstack((states['h1'], states['h2'],
                               states['h3'])).astype(int)
    # active_states = active_states.reshape(22, 1)
    active_states = active_states.reshape(H1+H2+H3, 1)

    weight_bias = np.append(weights, bias, axis=1)

    weight_bias_states = np.append(weight_bias, active_states, axis=1)

    #print(len(weight_bias_states))
    output_file = open(filename, 'wb')
    np.savetxt(output_file, weight_bias_states, delimiter=',')
    output_file.close()
    return filename


def interpretMnist(datasize,H1,H2,H3,model_path,train_data_path,label_format,stage,data):
    # main()
    D_in, D_out = datasize, 2

    model = PLNN(D_in, H1, H2, H3, D_out)

    model.load_state_dict(torch.load(model_path))
    # print(model)
    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset(train_data_path, label_format),
        batch_size=64, shuffle=True)

    model.eval()

    l = data.size

    data = data.reshape(-1, l)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    states, output = model(data,stage)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    inequality_coefficient_filename = './inequality.txt'
    calculate_inequality_cofficients_mnist(stage,H1,H2,H3,model, data, inequality_coefficient_filename)
    coefficients = change_inequality_signs(inequality_coefficient_filename)
    check_inequality_coefficients(data, inequality_coefficient_filename)
    return coefficients


def findSub(datasize,H1,H2,H3,model_path,train_data_path,label_format,stage, TSdata, data_index):
    length = len(TSdata)
    coefficients = interpretMnist(datasize,H1,H2,H3,model_path,train_data_path,label_format,stage, TSdata)
    weights = coefficients[:, :-2]
    bias = coefficients[:, length:-1]
    dict1 = {}

    # 向一个图片添加扰动并求出前k个重要区域，暂定k=50
    for x in range(0,34):
        for y in range(0,34):
            print("(",x,y,")")
            score = 0
            for i in range(0,10):
                new_ts = changeTSdata(TSdata, x, y)
                differ = calculate_product_1(weights, TSdata.reshape(length, 1), new_ts.reshape(length, 1), bias)
                score += differ
            score = score/10
            dict1[(x, y)] = score

    f1 = zip(dict1.values(), dict1.keys())
    c1 = sorted(f1, reverse=True)

    k = 60;
    list1 = []
    for i in range(0,k):
        list1.append(c1[i][1])
    return list1,data_index


def turn_to_picture(list,TSdata,picture_name):
    result = []
    for i in range(0,len(list)):
        locate_list = locate(list[i][0],list[i][1])
        result.extend(locate_list)

    change_array = np.zeros(shape=(36,36))
    for k in range(0, len(result)):
        row = int(result[k]/36)
        column = result[k]-36*row
        change_array[row][column] = 1

    TSdata.resize([36,36])
    new_picture = TSdata * change_array
    new_picture_array = np.array(new_picture)
    new_picture_array.resize([36,36])
    print(TSdata)
    print(new_picture_array)

    plt.subplot(121)
    plt.imshow(new_picture_array,cmap='Greys')
    plt.subplot(122)
    plt.imshow(TSdata,cmap='Greys')
    plt.savefig('C:\Pycharm\Projects\SDIP-Github\SDIP-Github\output' + '\\' + picture_name + '.jpg')
    # plt.savefig('C:\Pycharm\Projects\SDIP-Github\SDIP-Github\output'+'\\'+picture_name+'.jpg')
    plt.close()
    # plt.show()



# def generateMnistShapelets(train_data_path,class_label):
#     # 读入训练集的数据
#     train_data = pd.read_table(train_data_path, sep=',', header=None, engine='python').astype(float)
#     train_data_y = train_data.loc[:, 0].astype(int)
#     train_data_x = train_data.loc[:, 1:]
#     current_class_index = np.where(train_data_y == class_label)[0]
#
#     # 开始寻找shapelets
#     candidate=[]
#     for i in current_class_index:
#         one_ts = np.array(train_data_x.loc[i, :])
#         one_ts_label = np.array(train_data_y.loc[i])
#         one_ts_list = []
#
#         one_ts_list,index = findSub(784,4,16,2,'./Mnist_PLNN.pt',train_data_path,0,0,one_ts,i)


if __name__=='__main__':
    train_data = pd.read_table('./data/sc_test_grey.txt', sep=',', header=None, engine='python').astype(float)
    print("数据读入完成")
    train_data_x = train_data.loc[:, 1:]
    for i in range(0,1):
        one_ts = np.array(train_data_x.loc[i, :])
        list1, index = findSub(1296, 8, 16, 4, './PLNN_models/NeuSurface_models/Sc_PLNN_grey.pt', './data/sc_test_grey.txt', 0, 0, one_ts, 0)
        picture_name = "picture" + str(i)
        turn_to_picture(list1, one_ts, picture_name)