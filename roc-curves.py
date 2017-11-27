import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from skimage import io


class plot_rocs:
   
    def calcul_metrics(self, actual, predictions):
        cm = confusion_matrix(actual, predictions)
        print('Confusion Matrix:\n%s' %cm)
        print('Accuracy: %0.3f' %accuracy_score(actual, predictions))
        print('Precision: %0.3f' %precision_score(actual, predictions,average='binary',pos_label=1))
        print('Recall: %0.3f' %recall_score(actual, predictions,average='binary',pos_label=1))


    def plotar(self, actual, predictions):
        fpr, tpr, thresholds = roc_curve(actual, predictions, pos_label=1)
        roc_auc = auc(fpr, tpr)
        #Plot of a ROC curve for a specific class
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
	plt.savefig("teste.png")
        #plt.show()

pl = plot_rocs()

img1 = io.imread('binary_mask_person (3).bmp')
img2 = io.imread('dilate Person_3(Sin Gamma).bmp')
img1[img1==1] = 0;
img1[img1==2] = 1;
img1[img1==255] = 1;
img2[img2==1] = 0;
img2[img2==2] = 1;
img2[img2==255] = 1;

pl.calcul_metrics(img1.ravel(), img2.ravel())
pl.plotar(img1.ravel(), img2.ravel())

#plt.imshow(img1)
#pl.calcul_metrics([1, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 0, 0, 0])

