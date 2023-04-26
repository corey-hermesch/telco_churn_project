import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

# defining function to split train/validate/test into X and y dataframes AND return the baseline accuracy
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
def get_classifier_metrics(y_train, y_pred):
    """
    This function will
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

# defining a function to get accuracy scores of multiple LogisticRegression models
def get_multi_logit_scores(X_train, X_validate, y_train, y_validate):
    """
    This function will
    - take in X_train, X_validate, y_train, y_validate
    - make multiple Logistic Regression models with varying C values
        - C = [.01, .1, 1, 10, 100, 1000]
    - return dataframe with train_accuracy, validate_accuracy, and coefficients of each variable
    """
    C_values = [.01, .1, 1, 10, 100, 1000]
    results = []
    cols = ['C','train_acc','val_acc']
    coef_cols = ['coef_' + c for c in X_train.columns]
    results_df = pd.DataFrame(cols + coef_cols).T

    for x in C_values:
        logit = LogisticRegression(C=x)
        logit.fit(X_train, y_train)
        
        train_acc = logit.score(X_train, y_train)
        val_acc = logit.score(X_validate, y_validate)

        test = np.array([x, train_acc, val_acc])
        test_coef = logit.coef_
        combo_array = np.concatenate((test, test_coef[0]))
        new_df = pd.DataFrame(combo_array)
        results_df = pd.concat((results_df, new_df.T), axis=0)

    return results_df  

# defining a function to get accuracy scores of multiple knn models
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

# defining a function to get accuracy scores of multiple RandomForest models
def get_rf_scores(X_train, X_validate, y_train, y_validate):
    """
    This function will
    - take a while to run if you have a large dataset!
    - take in X_train, X_validate, y_train, y_validate
    - make multiple RandomForest classifier models with hyperparameters that vary:
        - max_depth varies from 1 to 10
        - min_samples_leaf varies from 1 to 10
    - returns a dataframe with train/validate accuracies and their difference
    """

    # initialize random forest accuracy dataframe
    rf_acc_init = pd.Series(range(1,11))
    rf_acc_df = pd.DataFrame(rf_acc_init, columns=['min_samples_leaf'])

    for y in range(1, 11): # max_depth = 1-10
        train_acc_list = []
        val_acc_list = []
        for x in range(1, 11):  # min_samples_leaf = 1-10
            rf = RandomForestClassifier(min_samples_leaf=x, random_state=42, max_depth = y, criterion='entropy')
            rf.fit(X_train, y_train)
            train_acc = rf.score(X_train, y_train)
            val_acc = rf.score(X_validate, y_validate)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
        new_col_t = 'trn_acc_depth_' + str(y)
        rf_acc_df[new_col_t] = pd.Series(train_acc_list)
        new_col_v = 'val_acc_depth_' + str(y)
        rf_acc_df[new_col_v] = pd.Series(val_acc_list)
        new_col_d = 'diff_' + str(y)
        rf_acc_df[new_col_d] = rf_acc_df[new_col_t] - rf_acc_df[new_col_v]
    
    return rf_acc_df

# defining a function to get accuracy scores of multiple Decision Tree models
def get_dtree_scores(X_train, X_validate, y_train, y_validate, crit='gini', max_d=10):
    """
    This function will
    - take in X_train, X_validate, y_train, y_validate (dataframes for modeling)
    - take in crit (criterion) with the default value 'gini'
        - other valid options are 'entropy' and 'log_loss'
    - take in max_d (max_depth) with default value of 10
        - this will set the number of trees to create with each having max_depth 1 to max_d
    - make 10 DecisionTree classifier models with max_depth from 1 to 10
    - return train and validate accuracies for each tree in a dataframe
    """
    #initialize results
    results=[]

    for x in range(1, max_d+1):
        tree = DecisionTreeClassifier(max_depth=x, criterion=crit)
        tree.fit(X_train, y_train)
        
        train_acc = tree.score(X_train, y_train)
        val_acc = tree.score(X_validate, y_validate)
        results.append([x, train_acc, val_acc])
            
    results_df = pd.DataFrame(results, index=(range(1,max_d+1)), columns=['max_depth','train_acc','val_acc'])
    results_df['difference'] = results_df.train_acc - results_df.val_acc
    return results_df  