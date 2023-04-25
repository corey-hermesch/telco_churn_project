### IMPORTS 

from env import host, user, password
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

np.random.seed(42)

### FUNCTIONS 

def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# generic function to get a sql pull into a dataframe
def get_mysql_data(sql_query, database):
    """
    This function will:
    - take in a sql query and a database (both strings)
    - create a connection url to mySQL database
    - return a df of the given query, connection_url combo
    """
    url = get_db_url(database)
    return pd.read_sql(sql_query, url)    

def get_csv_export_url(g_sheet_url):
    '''
    This function will
    - take in a string that is a url of a google sheet
      of the form "https://docs.google.com ... /edit#gid=12345..."
    - return a string that can be used with pd.read_csv
    '''
    csv_url = g_sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return csv_url

def get_telco_data(sql_query= """
                        SELECT  customer_id, gender, senior_citizen
                            , partner, dependents, tenure, phone_service
                            , multiple_lines, customers.internet_service_type_id
                            , internet_service_types.internet_service_type
                            , online_security, online_backup
                            , device_protection, tech_support
                            , streaming_tv, streaming_movies
                            , customers.contract_type_id
                            , contract_types.contract_type
                            , paperless_billing, customers.payment_type_id
                            , payment_types.payment_type
                            , monthly_charges, total_charges
                            , churn
                        FROM customers
                        JOIN contract_types USING (contract_type_id)
                        JOIN internet_service_types USING (internet_service_type_id)
                        JOIN payment_types USING (payment_type_id)
                    """
                    , filename="telco.csv"):
    
    """
    This function will:
    -input 2 strings: sql_query, filename 
        default query selects all columns from tables: customers, contract_types, internet_service_types, payment_types
        default filename "telco.csv"
    - check the current directory for filename (csv) existence
      - return df from that filename if it exists
    - If csv doesn't exist:
      - create a df of the sql_query
      - write df to csv
      - return that df
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        print ("csv file found and read")
        return df
    else:
        url = get_db_url('telco_churn')
        df = pd.read_sql(sql_query, url)
        df.to_csv(filename, index=False)
        print ("csv file not found, data read from sql query, csv created")
        return df
    
def prep_telco(df):
    """
    This function will
    - take in the telco_churn dataframe
    - clean it up (remove useless columns, rename some columns, and
      add encoded columns for categorical variables (columns) 
    - returns cleaned up (prepared) dataframe
    """
    df = df.drop(columns=['customer_id', 'internet_service_type_id', 'contract_type_id', 'payment_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    df['gender_encoded'] = df.gender.map({'Male': 0, 'Female': 1})
    df['partner_encoded'] = df.partner.map({'No': 0, 'Yes': 1})
    df['dependendents_encoded'] = df.dependents.map({'No': 0, 'Yes': 1})
    df['phone_service_encoded'] = df.phone_service.map({'No': 0, 'Yes': 1})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'No': 0, 'Yes': 1})
    df['churn_encoded'] = df.churn.map({'No': 0, 'Yes': 1})
    dummy_df = pd.get_dummies(df[['multiple_lines', 'internet_service_type', 'online_security', 'online_backup'
                               ,'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies'
                               ,'contract_type', 'payment_type']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco2(df):
    """
    This function will
    - take in the telco_churn dataframe
    - clean it up (remove useless columns, 
      remove unneeded columns noted in explore phase (CHANGE FROM prep_telco),
      rename some columns, and add encoded columns for categorical variables (columns) 
    - returns cleaned up (prepared) dataframe
    """
    drop_cols = ['customer_id', 'internet_service_type_id', 
                 'contract_type_id', 'payment_type_id',
                 'multiple_lines', 'phone_service', 
                 'streaming_tv', 'streaming_movies', 
                 'total_charges', 'gender', 'senior_citizen',
                 'partner', 'dependents']
    df = df.drop(columns=drop_cols)
#     df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
#     df['gender_encoded'] = df.gender.map({'Male': 0, 'Female': 1})
#     df['partner_encoded'] = df.partner.map({'No': 0, 'Yes': 1})
#     df['dependendents_encoded'] = df.dependents.map({'No': 0, 'Yes': 1})
#     df['phone_service_encoded'] = df.phone_service.map({'No': 0, 'Yes': 1})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'No': 0, 'Yes': 1})
    df['churn_encoded'] = df.churn.map({'No': 0, 'Yes': 1})
    dummy_cols = ['internet_service_type', 'online_security', 
                  'online_backup', 'device_protection', 'tech_support',
                  'contract_type', 'payment_type']
    dummy_df = pd.get_dummies(df[dummy_cols], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco_for_model(df):
    """
    This function will
    - take in telco dataframe from prep_telco function
    - remove non-encoded columns
    - remove unneccessary columns (repeated columns)
    - return df with only numeric columns ready for modeling
    """
    # getting rid of non-numeric columns to start the modeling phase
    drop_cols = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'internet_service_type',
                'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                'streaming_movies', 'contract_type', 'paperless_billing', 'payment_type', 'churn']
    # make "encoded" df with only the encoded columns for machine learning
    e_df = df.drop(columns=drop_cols)
    
    # phone_service was included in multiple_lines, so drop phone_service_encoded, too
    e_df = e_df.drop(columns=['phone_service_encoded'])
    
    # total_charges is directly related to tenure, so drop total_charges
    e_df = e_df.drop(columns=['total_charges'])
    
    # Since internet_service_type_None is repeated in several columns, I can delete them. 
    # A possibly better way is to encode them differently so the column names make more sense. maybe later
    repeated_cols = ['online_security_No internet service', 'online_backup_No internet service',
                    'device_protection_No internet service', 'tech_support_No internet service',
                    'streaming_tv_No internet service', 'streaming_movies_No internet service']
    e_df = e_df.drop(columns=repeated_cols)
    
    return e_df

def prep_telco_for_model2(df):
    """
    This function will
    - take in telco dataframe from prep_telco2 function
    - remove non-encoded columns
    - remove unneccessary columns (repeated columns)
    - return df with only numeric columns ready for modeling
    """
    # getting rid of non-numeric columns to start the modeling phase
    drop_cols = ['internet_service_type', 'online_security',
                 'online_backup', 'device_protection', 'tech_support',
                 'contract_type', 'paperless_billing', 'payment_type', 'churn']
    # make "encoded" df with only the encoded columns for machine learning
    e_df = df.drop(columns=drop_cols)
    
 
    # Since internet_service_type_None is repeated in several columns, I can delete them. 
    # A possibly better way is to encode them differently so the column names make more sense. maybe later
    repeated_cols = ['online_security_No internet service',
                     'online_backup_No internet service', 
                     'device_protection_No internet service',
                     'tech_support_No internet service']
    e_df = e_df.drop(columns=repeated_cols)
    
    return e_df

def split_function(df, target_var):
    """
    This function will
    - take in a dataframe (df) and a string (target_var)
    - split the dataframe into 3 data frames: train (60%), validate (20%), test (20%)
    -   while stratifying on the target_var
    - And finally return the three dataframes in order: train, validate, test
    """
    train, test = train_test_split(df, random_state=42, test_size=.2, stratify=df[target_var])
    
    train, validate = train_test_split(train, random_state=42, test_size=.25, stratify=train[target_var])

    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

def impute_feature(train, validate, test, feature='age', strat='median'):
    """
    This function will
    - take in train, validate, test dfs
    - take in a string which is the column name that has nan values
        -- default is 'age' (built off titanic df)
    - take in a string which is the strategy to impute values
        -- default is 'median'
    - impute nan values in the feature(age) column and fill with new values
    - return train, validate, test with imputed values
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
    imputer = imputer.fit(train[[feature]])
    train[[feature]] = imputer.transform(train[[feature]])
    validate[[feature]] = imputer.transform(validate[[feature]])
    test[[feature]] = imputer.transform(test[[feature]])
    
    return train, validate, test