# Title
 
# Project Description
 
Telco is a fictional telecommunications company akin to AT&T or Verizon. This project uses data science techniques to create a machine learning model that will predict customer churn, aka when a customer cancels their service with the company.
 
# Project Goal
 
* Discover drivers of customer churn at Telco
* Use drivers to develop a machine learning model to classify whether customers are likely to churn or not
* Customer churn is defined as a customer cancelling their service with the company
* This information could be used to intervene to prevent customer churn
 
# Initial Thoughts
 
My initial hypothesis is that monthly charges and tenure (length of time with the company) will have a larger impact on predicting customer churn than any of the other features of the data
 
# The Plan
 
* Aquire data from Codeup's SQL server
 
* Prepare data
   * Remove unneccessary columns and repeated columns.
   * Ensure remaining columns are in the correct format (e.g. floats vs strings for dollar amounts)
   * Search for and handle any null values
 
* Explore data in search of drivers of churn
   * Answer the following initial questions
       * How often does churn occur?
       * Does 'monthly_charges' affect churn?
       * Does 'tenure' affect churn?
       * Do any of the other features have a significant affect on churn?
      
* Develop a Model to predict if a customer will churn
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Type | Definition |
|:--------|:-----|:-----------|
|customer_id|string|unique identifer for each customer; format 1111-AAAAA|
|gender|string|Male or Female|
|senior_citizen|integer|1 for IS a senior citizen, 0 if not|
|partner|string|Yes if customer has a partner, No if not|
|dependents|string|Yes if customer has dependents, No if not|
|tenure|integer|number of months customer has been with Telco|
|phone_service|string|Yes if customer has phone service, No if not|
|multiple_lines|string|Yes, No, or No Phone Service|
|internet_service_type|string|Fiber optic, DSL, or None|
|online_security|string|Yes, No, or No internet service|
|online_backup|string|Yes, No, or No internet service|
|device_protection|string|Yes, No, or No internet service|
|tech_support|string|Yes, No, or No internet service|
|streaming_tv|string|Yes, No, or No internet service|
|streaming_movies|string|Yes, No, or No internet service|
|contract_type|string|Month-to-Month, One year, Two year|
|paperliss_billing|string|Yes if they have paperliss billing, No if not|
|payment_type|string|Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)|
|monthly_charges|float|This is the customer's monthly bill|
|total_charges|float|This is the total amount a customer has paid since being with the company|
|churn|string|Yes if customer churned, No if they stayed with company|

 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from (for example: [Kaggle](https://www.kaggle.com/datasnaek/chess) )
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Takeaway 1
* Takeaway 2...
 
# Recommendations
* Rec 1
* Rec 2 ...