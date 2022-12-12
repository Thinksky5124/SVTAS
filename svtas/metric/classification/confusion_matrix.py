'''
Author       : Thyssen Wen
Date         : 2022-11-22 15:19:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 23:10:19
Description  : file content
FilePath     : /SVTAS/svtas/metric/classification/confusion_matrix.py
'''
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os
import datetime
from ...utils.logger import get_logger
from ..base_metric import BaseMetric
from ..builder import METRIC

@METRIC.register()
class ConfusionMatrix(BaseMetric):
    """
        ref:https://blog.csdn.net/weixin_43760844/article/details/115208925 \\
        To visualize and caculate confusion matrix
    """
    def __init__(self,
                 actions_map_file_path: str,
                 img_save_path: str = None,
                 need_plot: bool =True,
                 need_color_bar: bool =True,
                 train_mode: bool = False):
        super().__init__()
        self.img_save_path = img_save_path
        self.need_color_bar = need_color_bar
        self.need_plot = need_plot
        self.train_mode = train_mode

        if self.img_save_path is not None:
            isExists = os.path.exists(self.img_save_path)
            if not isExists:
                os.makedirs(self.img_save_path)
        else:
            self.img_save_path = "./output"
        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.labels = dict()
        for a in actions:
            self.labels[int(a.split()[0])] = a.split()[1]
        self.matrix = np.zeros((len(self.labels), len(self.labels)))
        self.num_classes = len(self.labels)
    
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, vid, ground_truth_list, outputs):
        acc = 0.
        total = 1
        for labels, preds in zip(ground_truth_list, outputs['predict']):
            for p, t in zip(preds, labels):
                self.matrix[p, t] += 1
                if p != t:
                    total += 1
                elif p == t:
                    total += 1
                    acc += 1
        return acc / total

    def accumulate(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
		
		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        logger = get_logger("SVTAS")
        logger.info("Model performence in Classification task (Confusion Matrix): \n" + str(table))
        if self.plot and self.train_mode is False:
            self.plot(acc)
        
        # for next epoch
        self.reset()
        return acc

    def plot(self, acc):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), list(self.labels.values()), rotation=45)
        plt.yticks(range(self.num_classes), list(self.labels.values()))
        if self.need_color_bar:
            plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+str(acc)+')')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # note matrix[y, x] is not matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         fontsize=7,
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_save_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_confusion_matrix.png"), bbox_inches='tight', dpi=500)
        plt.close()