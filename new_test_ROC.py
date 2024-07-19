import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_roc_curves(model_types, dataset):
    if dataset == 'BOE':
        nb_classes = 3
    elif dataset == 'OCT':
        nb_classes = 4

    fpr_all = dict()
    tpr_all = dict()
    roc_auc_all = dict()
    precision_all = dict()
    recall_all = dict()

    for model_type in model_types:
        saved_path = './predictions'
        filename = '_'.join([dataset, model_type])
        filename = './predictions/ADDA_EM_BOE_to_CELL_iter4_predictions.pt'

        pred_result = torch.load(filename)
        print(pred_result.keys())

        y_true = pred_result['y_true_tgt']  # Update to 'y_true_tgt' or 'y_true_src' depending on the dataset
        y_pred = pred_result['y_pred_tgt']  # Update to 'y_pred_tgt' or 'y_pred_src' depending on the dataset
        y_prob = pred_result['y_prob_tgt']  # Update to 'y_prob_tgt' or 'y_prob_src' depending on the dataset

        y_true = np.array(y_true)  # Convert y_true to a NumPy array

        # Convert y_prob to a NumPy array if it's a list
        if isinstance(y_prob, list):
            y_prob = np.array(y_prob)

        # binarize the output
        y_true = label_binarize(y_true, classes=[i for i in range(nb_classes)])
        y_pred = label_binarize(y_pred, classes=[i for i in range(nb_classes)])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(nb_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= nb_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute precision and recall for each class
        precision = dict()
        recall = dict()
        for i in range(nb_classes):
            precision[i] = precision_score(y_true[:, i], y_pred[:, i])
            recall[i] = recall_score(y_true[:, i], y_pred[:, i])

        # Compute macro-average precision and recall
        precision_macro = sum(precision.values()) / nb_classes
        recall_macro = sum(recall.values()) / nb_classes

        print("Model:", model_type, " | Macro-Average AUC = {0:0.3f}".format(roc_auc["macro"]))
        print("Model:", model_type, " | Macro-Average Precision = {0:0.3f}".format(precision_macro))
        print("Model:", model_type, " | Macro-Average Recall = {0:0.3f}".format(recall_macro))

        fpr_all[model_type] = fpr["macro"]
        tpr_all[model_type] = tpr["macro"]
        roc_auc_all[model_type] = roc_auc["macro"]
        precision_all[model_type] = precision_macro
        recall_all[model_type] = recall_macro

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(8, 6))
    colors = cycle(['brown', 'deeppink', 'violet', 'darkviolet', 'slateblue', 'dodgerblue', 'c', 'lightgreen', 'orange', 'blue', 'green', 'red'])
    model_name = {'vgg_vat': 'Proposed', 'vgg': 'VGG-16+Pretrain', 'vgg_nopre': 'VGG-16', 'resnet18': 'Resnet+Pretrain',
                  'resnet18_nopre': 'Resnet', 'alexnet': 'AlexNet+Pretrain', 'alexnet_nopre': 'AlexNet', 'cnn-5': 'CNN',
                  'svm': 'SVM+HOG', 'pseudo': 'Pseudo-Labeling', 'tempens': 'Temporal Ensembling', 'meanteacher': 'Mean Teacher'}

    for model_type, color in zip(model_types, colors):
        plt.plot(fpr_all[model_type], tpr_all[model_type], color=color, lw=lw, label='{}'.format(model_name[model_type]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if dataset == 'BOE':
        title = 'ROC Curves on BOE Dataset'
    elif dataset == 'OCT':
        title = 'ROC Curves on CELL Dataset'

    plt.title(title)
    plt.legend(loc="lower right", fontsize=10)
    
    # Create directory if it doesn't exist
    output_dir = "./doc/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "ROC_" + dataset + "_add_semi_methods_prob.png"))
    plt.savefig(os.path.join(output_dir, "ROC_" + dataset + "_add_semi_methods_prob.eps"))
    plt.show()
    print("End")

model_types = ['svm', 'cnn-5', 'resnet18_nopre', 'resnet18', 'alexnet_nopre', 'alexnet', 'vgg_nopre', 'vgg', 'pseudo',
               'tempens', 'meanteacher', 'vgg_vat']

dataset = 'BOE'

plot_roc_curves(model_types, dataset)
