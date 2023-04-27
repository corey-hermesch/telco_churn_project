import numpy as np
import pandas as pd

import os

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

# defining a function to remove unneccessary columns discovered after explore phase
def prep_telco_for_model(df):
    """
    This function will
    - take in a telco dataframe from prep_telco_to_explore function
    - remove non-encoded columns
    - remove unneccessary columns (repeated columns)
    - remove columns I chose to eliminate in explore phase
    - return df with only numeric columns ready for modeling
    """
    
    features_to_keep = ['tenure', 'monthly_charges', 'contract_type_One year',
                        'contract_type_Two year', 
                        'payment_type_Credit card (automatic)', 
                        'payment_type_Electronic check',
                        'payment_type_Mailed check',
                        'paperless_billing_encoded',
                        'internet_service_type_Fiber optic',
                        'internet_service_type_None',
                        'online_security_Yes',
                        'online_backup_Yes',
                        'device_protection_Yes',
                        'tech_support_Yes', 'churn_encoded']
    
    # getting rid of non-numeric columns to start the modeling phase
    # getting rid of encoded columns that I chose to exclude after explore phase
    # Since internet_service_type_None is repeated in several columns, I can delete them. 
        
    return df[features_to_keep]

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

# defining a function to accept train/validate/test and return dataframes ready for modeling
def model_prep(train, validate, test, target):
    """
    This function will
    - accept train, validate, and test dataframes as well as the target variable
    - strip non-encoded columns
    - return X_train, X_validate, X_test, y_train, y_validate, y_test ready for modeling
    - lastly, returns baseline_accuracy
    """
    
    # remove unneccessary columns
    train = prep_telco_for_model(train)
    validate = prep_telco_for_model(validate)
    test = prep_telco_for_model(test)
    
    #split up train/validate/test on target and get baseline accuracy
    X_train, X_validate, X_test, y_train, y_validate, y_test, baseline_accuracy = (
        get_X_y_baseline(train, validate, test, target)
    )
    
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

# defining a function to print accuracy and recall for a y_train/validate/test and a y_pred
def print_model_metrics(y_split, y_pred, which):
    """
    This function will
    - accept y_split (y_train/validate/test), y_pred, and which dataset ('train','validate','test')
    - calculate and print accuracy and recall
    - return nothing
    """
    #calculate metrics
    conf = confusion_matrix(y_split, y_pred)
    TN, FP, FN, TP = conf.ravel()
    all_ = (TP + TN + FP + FN)
    accuracy = (TP + TN) / all_
    TPR = recall = TP / (TP + FN)

    # print y_split metrics
    print(f'Accuracy on {which} is {accuracy}')
    print(f'Recall on {which} is {recall}')

    return
    
# defining function for final_notebook to get Logistic Regression metrics for best logit model
def get_reg_all_features(X_train, X_validate, y_train, y_validate):
    """
    This function will
    - accept X_train, X_validate, y_train, y_validate
    - prints accuracy and recall for the best Logistic Regression model (C=1) for train and validate
    - returns nothing
    """
    # make and fit the model for train
    logit1 = LogisticRegression(C=1)
    logit1.fit(X_train, y_train)

    # get metrics for train
    y_pred1 = logit1.predict(X_train)
    print_model_metrics(y_train, y_pred1, 'train')
    print()
    
    # get metrics for validate
    y_pred2 = logit1.predict(X_validate)
    print_model_metrics(y_validate, y_pred2, 'validate')
    
    return

# defining a function for final_notebook to get best decision tree model with subset of features
def get_tree_features2(X_train, X_validate, y_train, y_validate):
    """ 
    This function will
    - accept X_train, X_validate, y_train, y_validate
    - print accuracy and recall for the best Logistic Regression model (C=1) for train and validate
    - returns nothing
    """
    features2 = ['monthly_charges', 'tenure', 'contract_type_One year', 'contract_type_Two year', 
                 'payment_type_Credit card (automatic)', 'payment_type_Electronic check', 
                 'payment_type_Mailed check', 'internet_service_type_Fiber optic', 
                 'internet_service_type_None', 'tech_support_Yes']
    # make and fit the model for train
    tree1 = DecisionTreeClassifier(max_depth=6)
    tree1.fit(X_train[features2], y_train)
    
    # get metrics for train
    y_pred1 = tree1.predict(X_train[features2])
    print_model_metrics(y_train, y_pred1, 'train')
    print()
    
    # get metrics for validate
    y_pred2 = tree1.predict(X_validate[features2])
    print_model_metrics(y_validate, y_pred2, 'validate')
    
    return
    
def get_knn_all_features(X_train, X_validate, y_train, y_validate):
    """
    This function will
    - accept X_train, X_validate, y_train, y_validate
    - prints accuracy and recall for the best knn model (n_neighbors=14) for train and validate
    - returns nothing
    """
    # make and fit the model for train
    knn1 = KNeighborsClassifier(n_neighbors=14)
    knn1.fit(X_train, y_train)

    # get metrics for train
    y_pred1 = knn1.predict(X_train)
    print_model_metrics(y_train, y_pred1, 'train')
    print()
    
    # get metrics for validate
    y_pred2 = knn1.predict(X_validate)
    print_model_metrics(y_validate, y_pred2, 'validate')
    
    return

# defining a function to write predictions.csv
def get_write_predictions_csv(X_train, X_test, y_train, customer_id_test):
    """
    This function will
    - accept X_train and y_train for developing a model
    - accept X_test to use to make predictions
    - accpet customer_id, a series with all the id's that match X_test
    - write a csv file in the current working directory with three columns: 
        - customer_id, probability of churn, prediction of churn (1=churn, 0=not_churn)
    """
    features2 = ['monthly_charges', 'tenure', 'contract_type_One year', 'contract_type_Two year', 
                 'payment_type_Credit card (automatic)', 'payment_type_Electronic check', 
                 'payment_type_Mailed check', 'internet_service_type_Fiber optic', 
                 'internet_service_type_None', 'tech_support_Yes']
    
    # make and fit the model for train
    tree1 = DecisionTreeClassifier(max_depth=6)
    tree1.fit(X_train[features2], y_train)
    
    # get probabilities/predictions for test
    y_prob1 = tree1.predict_proba(X_test[features2])
    y_pred1 = tree1.predict(X_test[features2])
    
    # turn into series for putting into dataframe
    prob_churn = pd.Series(y_prob1[:,1])
    churn_predict = pd.Series(y_pred1)
    
    # reset index of passed series customer_id_test so it will match prob_churn and churn_predict
    customer_ids = customer_id_test.reset_index().customer_id
    cols = ['customer_id', 'probability_of_churn', 'churn_predict']

    # concatenate the three series into a dataframe and fix the columns
    prediction_df = pd.concat([customer_ids, prob_churn, churn_predict], axis=1)
    prediction_df.columns = cols

    # write the df to a csv
    filename = 'predictions.csv'
    if os.path.isfile(filename):
        print ("csv file found and overwritten")
    else:
        print ("csv file not found; predictions.csv created")

    prediction_df.to_csv(filename)
    
    return