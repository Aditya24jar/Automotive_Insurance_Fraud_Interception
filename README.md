# Automotive Insurance Fraud Interception

## Overview
This project aims to detect insurance fraud in vehicular accidents using machine learning techniques. Our approach involves data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and deployment using Flask.


# Dataset Feature Description

| Feature               | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| *Month*             | Month of the accident                                                    |
| *WeekofMonth*       | Week of the accident                                                     |
| *DayofWeek*         | Day of the accident                                                      |
| *Make*              | Manufacture company of the car                                           |
| *AccidentArea*      | Area where the accident took place (Rural or Urban)                      |
| *DayofWeekClaimed*  | Day of filing the insurance claim                                        |
| *MonthClaimed*      | Month of filing the insurance claim                                      |
| *WeekofMonthClaimed*| Week of filing the insurance claim                                       |
| *Sex*               | Gender of the insurance claimant (Male or Female)                        |
| *MaritalStatus*     | Marital status of the claimant (Single, Married, Widow, Divorced)        |
| *Age*               | Age of the insurance policy holder                                       |
| *Fault*             | Responsible party (Policy holder or Third party)                         |
| *PolicyType*        | Type of policies issued                                                  |
| *VehicleCategory*   | Category of the vehicle (Sedan, Sport, Utility)                          |
| *VehiclePrice*      | Price of the vehicle                                                     |
| *PolicyNumber*      | Policy number issued to the customer                                     |
| *RepNumber*         | ID of the person who handles the claim                                   |
| *Deductible*        | Amount to be deducted before claim disbursement                          |
| *DriverRating*      | Experience rating of the driver                                          |
| *DaysPolicyAccident*| Days left in the policy since the accident                               |
| *DaysPolicyClaim*   | Days left in the policy since filing a claim                             |
| *PastNumberofClaims*| Number of past claims filed by the customer                              |
| *AgeofVehicle*      | Age of the vehicle (8 categories)                                        |
| *AgeofPolicyHolder* | Age of the policy holder (9 categories)                                  |
| *PoliceReportFiled* | Whether a police report was filed after the accident (Yes or No)         |
| *WitnessPresent*    | Whether a witness was present during the accident (Yes or No)            |
| *AgentType*         | Type of insurance agent (Internal or External)                           |
| *NumberofSupplements*| Number of supplements                                                   |
| *AddressChangeClaim*| Number of times change of address was requested by the insurance claimant|
| *NumberofCars*      | Number of cars owned by the claimant                                     |
| *Year*              | Year of the accident (1994, 1995, 1996)                                  |
| *BasePolicy*        | Base policy of the insurance (3 categories)                              |
| *FraudFound_P*      | Whether fraud was found (Binary Class: Yes or No)                        |



## Project Structure
1. Importing the necessary libraries
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - flask
    - scikit-learn
    - xgboost
    - optuna

2. Data Preprocessing and EDA
    - Outlier detection and missing value imputation.
    - Dropping unnecessary columns like Policy number and index.
    - Univariate analysis using bar charts, pie charts, and distribution plots.
    - Bivariate and multivariate analysis using violin plots, pair plots, contingency tables, and heatmaps.
    - Generated fraud_oracle_cleaned.csv after preprocessing.

3. Feature Engineering and Feature Selection
    - Read fraud_oracle_cleaned.csv.
    - Encoded categorical columns using dictionary mapping.
    - Analyzed correlations and dropped Base policy and Month columns.
    - Created a new dataset fraud_oracle_model_building.csv.
    - Built baseline models using Logistic Regression and XGBoost:
        - Logistic Regression baseline accuracy: 94%
        - But Accuracy was not a reliable metric here as dataset was highly imbalanced which is shown using Confusion Matrix

4. Final Model Building
    - Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
    - Trained the model using BalancedRandomForest, achieving appreciable results.
    - Fine-tuned XGBoost parameters using Optuna, achieving:
        - Final accuracy:  *91.82%*
        - Cross-validation score:  *96.49%*
        - These results indicate that the model has low bias and low variance.

5. Deployment using Flask
    - Created a pickle file to store model weights for the web application.
    - Created app.py and a templates folder for HTML files.
    - Loaded the model in Flask and created routes to request user input from index.html.
    - Applied necessary preprocessing to user input and predicted the output:
        - If predicted output is 1, return "Fraud Found".
        - If predicted output is 0, return "Fraud Not Found".
    - Added a requirements.txt file to the repository.

## Usage Instructions
### Prerequisites
- Python 3.7 or higher
- Install the required libraries:
    sh
    pip install -r requirements.txt
    
## Model Example
![Screenshot 2024-07-04 204654](https://github.com/Aryansh-kr/Automotive-Insurance-Fraud-Interception/assets/127012188/1842d3cb-2899-43e7-b8e4-8603d884393f)


### Running the Project
1. Clone the repository

2. Data Preprocessing and Model Building
    - Ensure you have the necessary CSV files (fraud_oracle_cleaned.csv and fraud_oracle_model_building.csv) in the project directory.
    - Run the Jupyter notebooks or Python scripts provided to preprocess the data and build the model.

3. Deployment
    - Run the Flask application:
        sh
        python app.py
        
    - Open your web browser and go to http://127.0.0.1:5000/.

4. Predicting Fraud
    - Enter the required information in the web form.
    - Submit the form to see the prediction (Fraud Found or Fraud Not Found).

## Repository Contents
- app.py: Flask application file.
- templates/: Folder containing HTML templates.
- model12.pkl: Pickle file containing the trained model.
- requirements.txt: File listing required libraries.
- Datasets/: Folder containing the CSV files used in the project.
