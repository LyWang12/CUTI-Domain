import os
import cv2
import pickle
import numpy as np
import torchvision.datasets as datasets


def get_mnist_data():
    list_img = []
    list_label = []
    data_size = 0

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    for i in range(len(mnist_trainset)):
        img = np.array(mnist_trainset[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

        list_img.append(img)
        list_label.append(np.eye(10)[mnist_trainset[i][1]])
        data_size += 1

    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]


def get_svhn_data():
    list_img = []
    list_label = []
    data_size = 0

    svhn_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=None)

    for i in range(len(svhn_trainset)):
        list_img.append(np.array(svhn_trainset[i][0]))
        assert list_img[-1].shape == (32, 32, 3)
        list_label.append(np.eye(10)[svhn_trainset[i][1]])
        data_size += 1

    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]


def get_usps_data():
    list_img = []
    list_label = []
    data_size = 0

    usps_trainset = datasets.USPS(root='./data', train=True, download=True, transform=None)

    for i in range(len(usps_trainset)):
        img = np.array(usps_trainset[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

        list_img.append(img)
        list_label.append(np.eye(10)[usps_trainset[i][1]])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]


def get_mnist_m_data():
    list_img = []
    list_label = []

    train_path = r'./data/mnistm_data.pkl'
    with open(train_path, 'rb') as f:
        obj = f.read()
    data = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}

    img_m = data['train']
    label_m = data['train_label']
    data_size = label_m.shape[0]  # 55000

    for i in range(data_size):
        list_img.append(img_m[i])
        list_label.append(np.eye(10)[label_m[i]])

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]


def get_cifar_data():
    list_img = []
    list_label = []
    data_size = 0
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    for i in range(len(cifar_trainset)):
        img = np.array(cifar_trainset[i][0])
        list_img.append(img)
        list_label.append(np.eye(10)[cifar_trainset[i][1]])
        data_size += 1

    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]


def get_stl_data():
    list_img = []
    list_label = []
    data_size = 0
    re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
    root = r'./data/stl10_binary'
    train_x_path = os.path.join(root, 'train_X.bin')
    train_y_path = os.path.join(root, 'train_y.bin')
    test_x_path = os.path.join(root, 'test_X.bin')
    test_y_path = os.path.join(root, 'test_y.bin')

    with open(train_x_path, 'rb') as fo:
        train_x = np.fromfile(fo, dtype=np.uint8)
        train_x = np.reshape(train_x, (-1, 3, 96, 96))
        train_x = np.transpose(train_x, (0, 3, 2, 1))
    with open(train_y_path, 'rb') as fo:
        train_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(train_y)):
        label = re_label[train_y[i] - 1]
        list_img.append(train_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1

    with open(test_x_path, 'rb') as fo:
        test_x = np.fromfile(fo, dtype=np.uint8)
        test_x = np.reshape(test_x, (-1, 3, 96, 96))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
    with open(test_y_path, 'rb') as fo:
        test_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(test_y)):
        label = re_label[test_y[i] - 1]
        list_img.append(test_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1


    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]


def get_visda_data(source):
    list_img = []
    list_label = []
    data_size = 0
    root_temp = r"/data1/WLY/code/CVPR/src/ntl/data/visda/{}".format(source)
    class_path = os.listdir(root_temp)
    for i in range(len(class_path)):
        class_temp = os.path.join(root_temp, class_path[i])
        img_path = os.listdir(class_temp)
        for j in range(1000):
            img_path_temp = os.path.join(class_temp, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            
            list_img.append(img)
            list_label.append(np.eye(12)[i])
            data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]



def get_augment_data(source):
    list_img = []
    list_label = []
    aug_data_size = 0
    gan_data_size = 0

    augment_trainset = os.listdir(r"D:/WLY/Documents/NUAA/TPAMI2/code/AdaIN/output/augment_{}".format(source))
    augment_labels = np.loadtxt(r"D:/WLY/Documents/NUAA/TPAMI2/code/AdaIN/output/augment_{}/labels".format(source))
    for i in range(len(augment_trainset) - 1):
        img = cv2.imread(r"D:/WLY/Documents/NUAA/TPAMI2/code/AdaIN/output/augment_{}/img_{}.png".format(source, i))
        list_img.append(img)
        list_label.append(np.eye(10)[int(augment_labels[i])])
        aug_data_size += 1
    print(aug_data_size)

    root_temp = r"D:/WLY/Documents/NUAA/TPAMI2/code/AdaIN/output/{}".format(source)
    class_path = os.listdir(root_temp)
    for i in range(len(class_path)):
        class_temp = os.path.join(root_temp, class_path[i])
        img_path = os.listdir(class_temp)
        for j in range(500):
            img_path_temp = os.path.join(class_temp, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            list_img.append(img)
            list_label.append(np.eye(len(class_path))[i])
            gan_data_size += 1
    print(gan_data_size)

    np.random.seed(0)
    ind = np.arange(aug_data_size+gan_data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    print(aug_data_size+gan_data_size)

    return [list_img, list_label, aug_data_size+gan_data_size]


def add_watermark(dataset_list, img_size=32, value=20):
    list_img, list_label, data_size = dataset_list

    mask = np.zeros(list_img[0].shape)

    for i in range(img_size):
        for j in range(img_size):
            if i % 2 == 0 or j % 2 == 0:
                mask[i, j, 0] = value
    img_list_len = list_img.shape[0]
    for i in range(img_list_len):
        list_img[i] = np.minimum(list_img[i].astype(int) + mask.astype(int), 255).astype(np.uint8)
    return [list_img, list_label, data_size]
