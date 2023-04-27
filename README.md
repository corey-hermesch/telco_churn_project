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
|paperless_billing|string|Yes if they have paperless billing, No if not|
|payment_type|string|Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)|
|monthly_charges|float|This is the customer's monthly bill|
|total_charges|float|This is the total amount a customer has paid since being with the company|
|churn|string|Yes if customer churned, No if they stayed with company|

 
# Steps to Reproduce
1) Clone this repo
2) Acquire the data from Codeup SQL server via code in this repo OR acquire from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and put the data into telco.csv in your working directory (cloned repo)
4) Run notebook
 
# Takeaways and Conclusions
* customers in the dataset churn ~26.5% of the time
* customers with higher monthly charges churn more often
    * monthly_charges >=30   => 31% churn rate
    * monthly_charges 70 - 110  => 36% churn ratee
    * in contrast, when monthly_charges < 30   => 10% churn rate
* customers with lower tenure churn more often
    * when tenure is <= 10/15/20 months, 50/46/44 % of people churned
    * in contrast, when tenure is > 20 months, only 14% of people churned
* other drivers of increased churn rate:
    * contract_type == Month-to-month
    * payment_type == Electronic check
    * internet_service_type == Fiber optic
    * online_security/online_backup/device_protection/tech_support == No

* The final model outperformed the baseline by a small margin (78% vs baseline of 73.5%) 
* However, the recall was only 48% on unseen data which means this model would miss 52% of customers who churn
* A very simple model that targeted customers with monthly_charges >= 30 would only be ~45% accurate, but it would capture ~91% of customers who are about to churn
* Another very simple model that targeted customers whose tenure is <= 20 would be ~69% accurate and capture ~68% of customers about to churn
 
# Recommendations
* Considering a variety of customer engagement strategies:
    * when the cost of engagement is low, utilize a simple model (monthly_charges >= 30 or tenure <= 20) to reach the max percentage of customers about to churn
    * when the cost of engagement is higher, consider the Decision Tree model since it's accuracy is higher, i.e. less likely to engage with customers who already do not plan to churn
* Consider collecting additional data to model. For example:
    * region and/or zip code,
    * number of customer service interactions (phone calls, service appointments), 
    * number of close contacts with Telco 