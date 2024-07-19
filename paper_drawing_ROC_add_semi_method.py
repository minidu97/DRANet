import os
import torch
from sklearn.metrics import precision_score, confusion_matrix, roc_curve,auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# model_types = ['vgg_vat', 'vgg', 'vgg_nopre', 'resnet18', 'resnet18_nopre', 'alexnet', 'alexnet_nopre', 'cnn-5','svm']

model_types = ['svm', 'cnn-5', 'resnet18_nopre', 'resnet18', 'alexnet_nopre', 'alexnet', 'vgg_nopre', 'vgg','pseudo','tempens','meanteacher','vgg_vat']

# model_type = 'alexnet_nopre'  # ['vgg_vat', 'vgg', 'vgg_nopre', 'resnet18', 'resnet18_nopre', 'alexnet', 'alexnet_nopre', 'cnn-5','svm']
dataset = 'BOE'         # ['BOE','OCT']

if dataset == 'BOE':
    nb_classes = 3
elif dataset == 'OCT':
    nb_classes = 4

fpr_all = dict()
tpr_all = dict()
roc_auc_all = dict()

for model_type in model_types:
    saved_path = './model_saved'
    filename = '_'.join([dataset,model_type])
    filename = os.path.join(saved_path,filename) + '_prediction.pt'

    pred_result = torch.load(filename)

    y_true = pred_result['y_true']
    y_pred = pred_result['y_pred']
    y_prob = pred_result['y_prob']

    # binarize the output
    y_true = label_binarize(y_true,classes=[i for i in range(nb_classes)])
    y_pred = label_binarize(y_pred,classes=[i for i in range(nb_classes)])

    # roc_curve:
    # y-axis: 真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # x-axis：假正率（False Positive Rate , FPR）

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        # fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print("Model:",model_type, " | Macro-Average AUC = {0:0.3f}".format(roc_auc["macro"]))

    fpr_all[model_type] = fpr["macro"]
    tpr_all[model_type] = tpr["macro"]
    roc_auc_all[model_type] = roc_auc["macro"]


# Plot all ROC curves
lw = 2
plt.figure(figsize=(8, 6))

# colors = cycle(['navy','aqua', 'darkorange', 'cornflowerblue'])
colors =cycle(['brown', 'deeppink', 'violet','darkviolet', 'slateblue','dodgerblue','c','lightgreen','orange','blue','green','red'])

model_name ={'vgg_vat':'Proposed',
             'vgg':'VGG-16+Pretrain',
             'vgg_nopre':'VGG-16',
             'resnet18':'Resnet+Pretrain',
             'resnet18_nopre':'Resnet',
             'alexnet':'AlexNet+Pretrain',
             'alexnet_nopre':'AlexNet',
             'cnn-5':'CNN',
             'svm':'SVM+HOG',
             'pseudo':'Pseudo-Labeling',
             'tempens':'Temporal Ensembling',
             'meanteacher':'Mean Teacher'}

for model_type, color in zip(model_types, colors):
    plt.plot(fpr_all[model_type], tpr_all[model_type], color=color, lw=lw,
             label='{}'.format(model_name[model_type]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

if dataset == 'BOE':
    title = 'ROC Curves on BOE Dataset'
elif dataset =='OCT':
    title = 'ROC Curves on CELL Dataset'

plt.title(title)
plt.legend(loc="lower right", fontsize=10)
plt.savefig("./doc/ROC_" + dataset + "_add_semi_methods_prob.png")
plt.savefig("./doc/ROC_" + dataset + "_add_semi_methods_prob.eps")
plt.show()
print("End")

