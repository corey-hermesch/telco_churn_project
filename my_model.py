import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# defining function to split train/validate/test into X and y dataframes AND return the baseline accuracy
# baseline accuracy code needs work. it's hard coded to 0 right now, and it should get the mode
def get_X_y_baseline(train, validate, test, target):
    """
    This function will
    - take the train, validate, and test dataframes as well as the target variable (string)
    - assumes train/validate/test are numeric columns only (i.e. ready for modeling)
    - split the dataframes into X_train/validate/test and y_train/validate/test
    - return all 6 dataframes and the baseline_accuracy rate
    """

    # set X_train/validate/test to be everything but the target
    X_train = train.drop(columns=[target])
    X_validate = validate.drop(columns=[target])
    X_test = test.drop(columns=[target])

    # set y_train/validate/test to be only the target
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    # Set baseline accuracy 
    
    baseline_accuracy = (train[target] == train[target].mode()[0]).mean()
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test, baseline_accuracy

# defining a function to get metrics for a set of predictions vs a train series
# need to study a bit on what ravel does. which thing does it use for what the positive case is? 0 or 1?
def get_tree_metrics(y_train, y_pred):
    """
    This functiion will
    - take in a y_train series and a y_pred (result from a classifier.predict)
    - assumes just two options for confusion matrix, i.e. not 3 or more categories (I think)
    - prints out confusion matrix with row/column labeled with actual/predicted and the unique values in y_train
    - returns TN, FP, FN, TP (# of True Negatives, False Positives, False Negatives, True Positives from confusion matrix
    """
    print("CONFUSION MATRIX")
    conf = confusion_matrix(y_train, y_pred)
    print(pd.DataFrame(
          conf,
          index=[label.astype(str) + '_actual' for label in sorted(y_train.unique())],
          columns=[label.astype(str) + '_predicted' for label in sorted(y_train.unique())])
        )
    print()
    print("Classification Report:")
    print(classification_report(y_train, y_pred))
    
    TN, FP, FN, TP = conf.ravel()

    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN
    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")
    
    return TN, FP, FN, TP

def get_knn_metrics(X_train, X_validate, y_train, y_validate, weights_='uniform', max_n=20):
    """
    This function will
    - take in X_train, X_validate, y_train, y_validate, weights_, max_n
      -- weights_: default 'uniform', only other option is 'distance'
      -- max_n: default 20 - max number of neighbors to try
    - get train_accuracy and validate_accuracy for n_neighbors = 1-20
    - return dataframe with results where the index == n_neighbors (1-20)
    """
    results = []
    for i in range(1,max_n+1):
        knn = KNeighborsClassifier(n_neighbors=i, weights=weights_)
        knn.fit(X_train, y_train)
        train_acc = knn.score(X_train, y_train)
        val_acc = knn.score(X_validate, y_validate)
        results.append([train_acc, val_acc])

    results_df = pd.DataFrame(results, index=(range(1,max_n+1)), columns=['train_acc', 'val_acc'])
    return results_df