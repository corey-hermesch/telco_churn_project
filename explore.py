import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def cat_hist_plot(df, cols, target):
    """
    This function was made specifically for telco, but should be generic for
    any time one wants to plot a series of categorical variables via sns histplots.
    - takes in
        -- df: dataframe with data to visualize
        -- cols: list of columns that are categorical variables
        -- target: string with the name of the target column
    - returns nothing
    - displays histplots of each column in cols on the x axis and target on the y axis
    """
    target_values = df[target].value_counts()
    for col in cols:
        for i in range(0,target_values.size):
            sns.histplot(df[df[target] == target_values.index[i]][col])
        plt.title(f'histplot of {col} vs {target}')
        plt.show()  

def get_telco_ne_df(df_telco):
    # ne is short for no_encoding. removing the encoded columns will just make it a little easier to look at
    cols = np.array(df_telco.columns[0:21])
    return df_telco[cols]

def get_hist_churn(df, target='churn'):
    """
    This function will
    - take in a dataframe with a target (default target = churn)
    - print a histplot of the target variable with a title specific to telco
    """
    s = df[target].value_counts(normalize=True)
    percent = round(s.loc['Yes'], 3)
    t = f'Customers who churn represent ~{percent*100}% of the data' 
    sns.histplot(data=df, x='churn', hue='churn')
    plt.title(t)
    plt.show()
    
def get_box_plot(df, target='churn', feature='monthly_charges'):
    """
    This function will
    - take in a dataframe, a target, and a feature
    - print a boxplot of the feature vs the target
    - default target and feature are specific to telco database
    
    """   
    sns.boxplot(data=df, x=target, y=feature)
    plt.show()

def get_hist_plot(df, target='churn', feature='monthly_charges'):
    """
    This function will
    - take in a dataframe, a target with two values, and a feature
    - print a histplot of the feature vs the target
    - default target and feature are specific to telco database
    
    """
    # print histplot
    s = list(df[target].value_counts().sort_values().index)
    blue_label = f'no {target}'
    red_label = target
    sns.histplot(df[df[target] == s[1]][feature], color='blue', label=blue_label)
    sns.histplot(df[df[target] == s[0]][feature], color='red', label=red_label)

    plt.title(f'histplot of {feature}:  red = {target}, blue = no {target}')
    plt.legend()
    
    if feature == 'payment_type':
        plt.xticks(rotation=20)
    plt.show()
    
def get_mannwhitneyu(df, target='churn', feature='monthly_charges'):
    """
    This function will
    - take in the telco dataframe, target (default='churn'), feature (default='monthly_charges')
    - print the t, p values from the mannwhitneyu stats test (for a categorical var vs a cotinuous variable)
    - returns nothing
    """
    # separate feature column into No's and Yes's
    no_df = df[df[target] == 'No'][feature]
    yes_df = df[df[target] == 'Yes'][feature]    

    # do stats test 
    t, p = stats.mannwhitneyu(no_df, yes_df)
    print (f't = {t}')
    print (f'p = {p}')
    
def get_chi_results(df, target='churn', feature='contract_type'):
    """
    This function will
    - take in a dataframe with at least two columns, the target, and a feature
    - both the target and feature must be categorical variables
    - defaults: target='churn', feature='contract_type'
    - prints out the chi^2 and p values
    - returns nothing
    """

    # make contingency table
    observed = pd.crosstab(df[target], df[feature])
    
    # do stats test and print results
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print (f'chi^2 = {chi2}')
    print (f'p     = {p}')