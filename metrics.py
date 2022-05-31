import numpy as np
from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    #print(preds)
    #print(labels)
    #exit()
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels, label_to_id):
    label_dict_inverse = {v: k for k, v in label_to_id.items()}

    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')



def f1_score_func_QA(preds, labels):
    #print(preds)
    #print(labels)
    #exit()
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class_QA(preds, labels, id_to_label):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()

    print(id_to_label)
    print(labels)

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {id_to_label[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')
