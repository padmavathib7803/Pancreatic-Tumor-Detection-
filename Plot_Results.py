from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def Plot_ROC_Curve():
    cls = ['CNN-BiLSTM', 'ResNet-50', 'CNN', 'BiGRU', 'EDRI-GBOA-ADBiGRU']
    for a in range(No_of_Dataset):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset - ' + str(a + 1) + ' - ROC Curve')
        colors = ["blue", "darkorange", "limegreen", "deeppink", "black"]
        Y_Score = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)
        for i, color in enumerate(colors):  # For all classifiers
            Predicted = Y_Score[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'NGO-ADBiGRU', 'ROA-ADBiGRU', 'SRO-ADBiGRU', 'GBOA-ADBiGRU', 'EDRI-GBOA-ADBiGRU']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure(facecolor='#f0f0f0')
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='NGO-ADBiGRU')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='ROA-ADBiGRU')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='SRO-ADBiGRU')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='GBOA-ADBiGRU')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EDRI-GBOA-ADBiGRU')
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def plot_Graph():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_ACT.npy', allow_pickle=True)
    # eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']

    Algorithm = ['NGO-ADBiGRU', 'ROA-ADBiGRU', 'SRO-ADBiGRU', 'GBOA-ADBiGRU', 'EDRI-GBOA-ADBiGRU']
    Classfier = ['CNN-BiLSTM', 'ResNet-50', 'CNN', 'BiGRU', 'EDRI-GBOA-ADBiGRU']
    Graph_Term = [0, 1, 4, 7, 10]
    # plt.style.use('fivethirtyeight')
    for i in range(No_of_Dataset):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            # plt.style.use("cyberpunk")
            fig = plt.figure(figsize=(10, 6))
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
            ax.set_axisbelow(True)
            X = np.arange(Graph.shape[0])
            barWidth = 0.15
            plt.gca().yaxis.grid(True, which="major", linewidth=1)
            color = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', '#ee6055']
            for l in range(Graph.shape[1] - 5):

                if l % 2 == 0:

                    bars = ax.bar(X + (l * barWidth * 1.15), Graph[:, l], color=color[l - 5], width=barWidth, edgecolor='#032b43',
                                  label=Algorithm[l - 5])
                    # Add text annotations above the bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.01),  # Adjust 2% above the bar height
                                 f"{int(np.round(height, 1))}", ha='center', va='bottom', fontsize=10, weight='bold', rotation=0)

                else:
                    bars = ax.bar(X + (l * barWidth * 1.15), Graph[:, l], color=color[l - 5], width=barWidth, edgecolor='#032b43',
                                  label=Algorithm[l - 5])
                    # Add text annotations above the bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, height * 0.01,  # Adjust 2% above the bar height
                                 f"{int(np.round(height, 1))}", ha='center', va='bottom', fontsize=10, weight='bold', rotation=0)

            plt.xticks(X + (((len(Algorithm)) * barWidth) / 2.2), ('ReLU', 'Linear', 'Tanh', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Function')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Term[j]])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=c, markersize=8) for c
                           in color]
            plt.legend(dot_markers, Algorithm, bbox_to_anchor=(1.0, 0.4), fontsize=8,
                       frameon=False, ncol=1)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            ax.grid(which='major', axis='y', linestyle='-')
            path1 = "./Results/Dataset_%s_Act_Alg_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            # -----------------------------------------------Classifier------------------------------------------------

            fig = plt.figure(figsize=(10, 6))
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
            ax.set_axisbelow(True)
            X = np.arange(Graph.shape[0])
            barWidth = 0.15
            plt.gca().yaxis.grid(True)

            coloru = ['#ff9b85', '#60d394', '#aaf683', '#ffd97d', '#ee6055']
            for l in range(5, Graph.shape[1]):
                if l % 2 == 0:
                    bars = ax.bar(X + (l * barWidth * 1.15), Graph[:, l], color=coloru[l - 5], width=barWidth, edgecolor='#032b43',
                                  label=Classfier[l - 5])

                    # Add text annotations above the bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.01),  # Adjust 2% above the bar height
                                 f"{int(np.round(height, 1))}", ha='center', va='bottom', fontsize=10, weight='bold', rotation=0)

                else:
                    bars = ax.bar(X + (l * barWidth * 1.15), Graph[:, l], color=coloru[l - 5], width=barWidth, edgecolor='#032b43',
                                  label=Classfier[l - 5])

                    # Add text annotations above the bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, height * 0.01,  # Adjust 2% above the bar height
                                 f"{int(np.round(height, 1))}", ha='center', va='bottom', fontsize=10, weight='bold', rotation=0)

            plt.xticks(X + (((len(Algorithm)) * barWidth) / 0.6), ('ReLU', 'Linear', 'Tanh', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Function')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Term[j]])
            dot_marker = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=c, markersize=8) for c
                          in coloru]
            plt.legend(dot_marker, Classfier, bbox_to_anchor=(1.0, 0.4), fontsize=8,
                       frameon=False, ncol=1)

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            ax.grid(which='major', axis='y', linestyle='-')
            path1 = "./Results/Dataset_%s_Act_Met_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Table():
    eval = np.load('Eval_all_KFOLD.npy', allow_pickle=True)
    Algorithm = ['Kfold', 'NGO-ADBiGRU', 'ROA-ADBiGRU', 'SRO-ADBiGRU', 'GBOA-ADBiGRU', 'EDRI-GBOA-ADBiGRU']
    Classifier = ['Kfold', 'CNN-BiLSTM', 'ResNet-50', 'CNN', 'BiGRU', 'EDRI-GBOA-ADBiGRU']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 3, 8, 10]).astype(int)
    Table_Terms = [0, 3, 8, 10]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = ['Kfold 1', 'Kfold 2', 'Kfold 3', 'Kfold 4', 'Kfold 5']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Kfold)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('------------------------------- Dataset - ', i + 1, table_terms[k], ' Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset -', i + 1, table_terms[k], ' Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Confusion():
    for n in range(No_of_Dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)
        class_1 = ['Benign', 'Malignant', 'Localized', 'Regional', 'Metastatic']
        class_2 = ['Benign', 'Malignant', 'Localized', 'Regional', 'Metastatic']
        class_3 = ['Benign', 'Malignant', 'Localized', 'Regional', 'Metastatic']
        Classes = [class_1, class_2, class_3]

        confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=Classes[n])
        cm_display.plot()
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.yticks(rotation=45)
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    Plot_ROC_Curve()
    plotConvResults()
    plot_Graph()
    Table()
    Plot_Confusion()
