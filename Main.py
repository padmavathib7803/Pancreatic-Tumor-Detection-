import os
import cv2 as cv
import nibabel as nib
import pydicom as dicom
from numpy import matlib
from tqdm import tqdm

from EGBOA import EGBOA
from GBOA import GBOA
from Global_Vars import Global_Vars
from Model_Feat_ViT import Model_Region_Vision_Transformer
from NGO import NGO
from Obj_Fun import Obj_fun_CLS
from Plot_Results import *
from ROA import ROA
from SRO import SRO

no_dataset = 3

# Read Dataset 1
an = 0
if an == 1:
    Dir = './Dataset/Dataset1/Images/Images/'
    Images = []
    List_Dir = os.listdir(Dir)
    for i in range(len(List_Dir)):
        print(i, len(List_Dir))
        Img = Dir + List_Dir[i]
        Image = cv.imread(Img)
        Imag = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
        Imagee = cv.resize(Imag, (512, 512))
        Images.append(Imagee)
    np.save('Img_1.npy', Images)

# Generate Target for Dataset 1
an = 0
if an == 1:
    Dir = './Dataset/Dataset1/Masks/Masks/'
    Target = []
    Counts = []
    List_Dir = os.listdir(Dir)
    for i in range(len(List_Dir)):
        Img = Dir + List_Dir[i]
        Image = cv.imread(Img)
        Image[Image != 0] = 255
        Count = np.count_nonzero(Image)
        Counts.append(Count)
        # print(i, Count)
        if 0 == Count:
            Target.append(0)
        elif 0 < Count <= 500:
            Target.append(1)
        elif 500 < Count <= 1000:
            Target.append(2)
        elif 1000 < Count <= 1500:
            Target.append(3)
        elif Count > 1500:
            Target.append(4)

    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1

    np.save('Tar_1.npy', target)

# shuffle dataset 1
an = 0
if an == 1:
    Target = np.load('Tar_1.npy', allow_pickle=True)
    Images = np.load('Img_1.npy', allow_pickle=True)
    index = np.arange(len(Images))
    np.random.shuffle(index)
    Org_Img = np.asarray(Images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Image_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Read dataset 2
an = 0
if an == 1:
    DirectoryImage = './Dataset/Dataset2/Task07_Pancreas/imagesTr/'
    DirectoryLabel = './Dataset/Dataset2/Task07_Pancreas/labelsTr/'
    List_dir = os.listdir(DirectoryImage)
    List_lab = os.listdir(DirectoryLabel)

    Images1 = []
    Labels1 = []
    Counts = []
    Target = []

    count = 0
    iter = 0
    while True:
        if count == 10:
            break
        else:
            filename = DirectoryImage + List_lab[iter]
            filename1 = DirectoryLabel + List_lab[iter]
            if List_dir[iter][0] == '.':
                pass
            else:
                img = nib.load(filename)
                lab = nib.load(filename1)
                Image = img.get_fdata()
                Label = lab.get_fdata()
                for j in range((img.shape[2])):
                    # for k in range(img.shape[3]):
                    print(iter, j)
                    image = Image[:, :, j].astype(np.uint8)
                    label = (Label[:, :, j] * 255).astype(np.uint8)
                    Images1.append(image)
                    Labels1.append(label)
                count += 1
            iter += 1

    for k in range(len(Labels1)):
        Lab = Labels1[k]
        Lab[Lab != 0] = 255
        Count = np.count_nonzero(Lab)
        Counts.append(Count)
        print(k, Count)
        if 0 == Count:
            Target.append(0)
        elif 0 < Count <= 1000:
            Target.append(1)
        elif 1000 < Count <= 2000:
            Target.append(2)
        elif 2000 < Count <= 3000:
            Target.append(3)
        elif Count > 3000:
            Target.append(4)

    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Images1))
    np.random.shuffle(index)
    Datas = np.asarray(Images1)
    Targ = np.asarray(target)
    Shuffled_Datas = Datas[index]
    Shuffled_Target = Targ[index]
    np.save('Image_2.npy', np.asarray(Shuffled_Datas))
    np.save('Target_2.npy', np.asarray(Shuffled_Target))

# Read Dataset 3
an = 0
if an == 1:
    # Read Images
    Image = []
    path = './Dataset/Dataset3/manifest-1599750808610/Pancreas-CT'
    in_dir = os.listdir(path)
    in_dir.remove('LICENSE')
    for i in tqdm(range(len(in_dir))):
        out_dir = path + '/' + in_dir[i] + '/'
        in_file = os.listdir(out_dir)
        for j in range(len(in_file)):
            file = out_dir + '/' + in_file[j]
            List_file = os.listdir(file)
            for k in range(len(List_file)):
                Sub_file = file + '/' + List_file[k] + '/'
                List_Sub = os.listdir(Sub_file)
                for l in range((len(List_Sub)) // 5):
                    img = Sub_file + List_Sub[l]
                    ds = dicom.dcmread(img)
                    image = ds.pixel_array.astype('uint8')
                    Image.append(image)

    # Read Mask
    Mask = []
    paths = './Dataset/Dataset3/TCIA_pancreas_labels-02-05-2017/'
    in_dir = os.listdir(paths)
    for i in tqdm(range(len(in_dir))):
        File = paths + in_dir[i]
        lab = nib.load(File)
        Label = lab.get_fdata()
        for j in range((Label.shape[2]) // 2):
            # print(len(in_dir), j)
            label = (Label[:, :, j] * 255).astype(np.uint8)
            Mask.append(label)

    # Shuffle Data and Mask
    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Org_msk = np.asarray(Mask)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Org_msk[index]

    # Generate Target from Mask
    Counts = []
    Target = []
    for x in tqdm(range(len(Shuffled_Target))):
        label = Shuffled_Target[x]
        label[label != 0] = 255
        Count = np.count_nonzero(label)
        Counts.append(Count)
        if Count == 0:
            Target.append(0)
        elif 1 <= Count <= 400:
            Target.append(1)
        elif 400 < Count <= 800:
            Target.append(2)
        elif 800 < Count <= 1500:
            Target.append(3)
        elif Count > 1500:
            Target.append(4)
    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Image_3.npy', Shuffled_Datas[:2500, :, :])
    np.save('Target_3.npy', target[:2500, :])

# Feature Extraction
an = 1
if an == 1:
    for n in range(no_dataset):
    # for n in range(1, 2):
        data = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)#[:10, :, : ]
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)#[:10, :]
        Feature1 = Model_Region_Vision_Transformer(data, Target)  # Region Vision transformer
        # Feature2 = Model_VGG19(data)  # Visual Geometry Group-19
        # Feature3 = Model_FPN(data)  # Feature Pyramid Network
        np.save('Feature_RViT_' + str(n + 1) + '.npy', Feature1)
        # np.save('Feature_VGG19_' + str(n + 1) + '.npy', Feature1)
        # np.save('Feature_FPN_' + str(n + 1) + '.npy', Feature1)


# optimal feature selection
an = 0
if an == 1:
    for n in range(no_dataset):
        Feat = np.load('Rbm_Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Dataset
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat(np.append((1 * np.ones((1, Chlen - 100))), (0.01 * np.ones((1, Chlen - 100)))), Npop, 1)
        xmax = matlib.repmat(np.append((Feat.shape[1] * np.ones((1, Chlen - 100))), (0.09 * np.ones((1, Chlen - 100)))),
                             Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = Obj_fun_CLS
        max_iter = 50

        print('NGO....')
        [bestfit1, fitness1, bestsol1, Time1] = NGO(initsol, fname, xmin, xmax, max_iter)  # NGO

        print('ROA....')
        [bestfit2, fitness2, bestsol2, Time2] = ROA(initsol, fname, xmin, xmax, max_iter)  # ROA

        print('SRO....')
        [bestfit3, fitness3, bestsol3, Time3] = SRO(initsol, fname, xmin, xmax, max_iter)  # SRO

        print('BOA....')
        [bestfit4, fitness4, bestsol4, Time4] = GBOA(initsol, fname, xmin, xmax, max_iter)  # GBOA

        print('GBOA....')
        [bestfit5, fitness5, bestsol5, Time5] = EGBOA(initsol, fname, xmin, xmax, max_iter)  # EGBOA

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BEST_Sol_' + str(n + 1) + '.npy', BestSol)

# Feature Selection
an = 0
if an == 1:
    for n in range(no_dataset):
        Feat1 = np.load('RViT_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat2 = np.load('VGG19_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('FPN_Feature_' + str(n + 1) + '.npy', allow_pickle=True)

        Feat = [Feat1, Feat2, Feat3]
        bests = np.load('BEST_Sol_' + str(n + 1) + '.npy', allow_pickle=True).astype(int)
        sol = np.round(bests[4, :]).astype(np.int16)
        feat = []
        for i in range(len(Feat)):
            Feat = Feat[i]
            for j in range(Feat.shape[0]):
                F1 = Feat[j].astype('f')
                feat_1 = sol[100:] * F1[:, sol[:100]]
                feat.append(feat_1)
            np.save('Selected_Feature_' + str(n + 1) + '.npy', feat)

# Classification
an = 0
if an == 1:
    for m in range(no_dataset):
        Feature = np.load('Selected_Feature_' + str(m + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(m + 1) + '.npy', allow_pickle=True)  # loading step
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((5, 14))
            Feat = Feature
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[0, :], pred = Model_MobileNet(Train_Data, Train_Target, Test_Data,  Test_Target)  # Model MobileNet
            Eval[1, :], pred1 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model LSTM
            Eval[2, :], pred2 = Model_DTCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DTCN
            Eval[3, :], pred3 = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model Bi-LSTM
            Eval[4, :], pred4 = Model_DBiGRU(Train_Data, Train_Target, Test_Data, Test_Target)  # DTCN + LSTM
            eval.append(Eval)
    np.save('Eval_all.npy', eval)  # Save Eval


Plot_ROC_Curve()
plotConvResults()
plot_Graph()
Table()
Plot_Confusion()