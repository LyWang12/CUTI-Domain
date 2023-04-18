import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

IMAGE_SIZE = 64   # dight, cifar-stl
# IMAGE_SIZE = 224   # office-home, vsida2017


dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

dataTransform_aug = transforms.Compose([
    transforms.ToTensor()
])
    
class Cus_Dataset(data.Dataset):
    def __init__(self, mode=None, dataset_1=None, begin_ind1=0, size1=0,
                                  dataset_2=None, begin_ind2=0, size2=0,
                                  dataset_3=None, begin_ind3=0, size3=0,
                                  dataset_4=None, begin_ind4=0, size4=0):

        self.mode = mode
        self.list_img = []
        self.list_img1 = []
        self.list_img2 = []
        
        self.list_label = []
        self.list_label1 = []
        self.list_label2 = []

        self.data_size = 0
        self.transform = dataTransform

        if self.mode == 'train':

            self.data_size = size1

            path_list1 = dataset_1[0][begin_ind1: begin_ind1+size1]
            path_list2 = dataset_2[0][begin_ind2: begin_ind2+size2]

            for file_path in path_list1:
                 self.list_img1.append(file_path)

            for file_path in path_list2:
                self.list_img2.append(file_path)


            self.list_label1 = dataset_1[1][begin_ind1: begin_ind1+size1]
            self.list_label2 = dataset_2[1][begin_ind2: begin_ind2+size2]

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img1 = np.asarray(self.list_img1)
            self.list_img1 = self.list_img1[ind]
            self.list_img2 = np.asarray(self.list_img2)
            self.list_img2 = self.list_img2[ind]

            self.list_label1 = np.asarray(self.list_label1)
            self.list_label1 = self.list_label1[ind]
            self.list_label2 = np.asarray(self.list_label2)
            self.list_label2 = self.list_label2[ind]

        elif self.mode == 'author':

            self.data_size = size1

            path_list1 = dataset_1[0][begin_ind1: begin_ind1+size1]
            path_list2 = dataset_2[0][begin_ind2: begin_ind2+size2]
            path_list3 = dataset_3[0][begin_ind3: begin_ind3+size3]
            path_list4 = dataset_4[0][begin_ind4: begin_ind4+size4]
            label_list1 = dataset_1[1][begin_ind1: begin_ind1+size1]
            label_list2 = dataset_2[1][begin_ind2: begin_ind2+size2]
            label_list3 = dataset_3[1][begin_ind3: begin_ind3+size3]
            label_list4 = dataset_4[1][begin_ind4: begin_ind4+size4]


            for i in range(size1):
                self.list_img1.append(path_list1[i])
                self.list_label1.append(label_list1[i])

            for i in range(size2):
                self.list_img2.append(path_list2[i])
                self.list_label2.append(label_list2[i])

            for i in range(size3):
                self.list_img2.append(path_list3[i])
                self.list_label2.append(label_list3[i])

            for i in range(size4):
                self.list_img2.append(path_list4[i])
                self.list_label2.append(label_list4[i])

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img1 = np.asarray(self.list_img1)
            self.list_img1 = self.list_img1[ind]
            self.list_img2 = np.asarray(self.list_img2)
            self.list_img2 = self.list_img2[ind]

            self.list_label1 = np.asarray(self.list_label1)
            self.list_label1 = self.list_label1[ind]
            self.list_label2 = np.asarray(self.list_label2)
            self.list_label2 = self.list_label2[ind]

        elif self.mode == 'val':

            self.data_size = size1

            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]

            for file_path in path_list:
                self.list_img.append(file_path)

            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]


    def __getitem__(self, item):
        if self.mode == 'train':
            img1 = self.list_img1[item]
            img2 = self.list_img2[item]
            label1 = self.list_label1[item]
            label2 = self.list_label2[item]
            return self.transform(img1), torch.LongTensor(label1), self.transform(img2), torch.LongTensor(label2)
        elif self.mode == 'author':
            img1 = self.list_img1[item]
            img2 = self.list_img2[item]
            label1 = self.list_label1[item]
            label2 = self.list_label2[item]
            return self.transform(img1), torch.LongTensor(label1), self.transform(img2), torch.LongTensor(label2)
        elif self.mode == 'val':
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])


    def __len__(self):
        return self.data_size


