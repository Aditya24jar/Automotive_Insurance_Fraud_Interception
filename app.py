from flask import Flask, request, jsonify, render_template
import xgboost
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model12.pkl', 'rb'))


# Assuming the StandardScaler is fitted on the training data
scaler = StandardScaler()

# Preprocessing mappings
map_month = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

map_day = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
    'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

AccidentArea_map = {'Urban': 0, 'Rural': 1}

Manufacture_company_map = {
    'Pontiac': 0, 'Toyota': 1, 'Honda': 2, 'Mazda': 3, 'Chevrolet': 4,
    'Accura': 5, 'Ford': 6, 'VW': 7, 'Dodge': 8, 'Saab': 9, 'Mercury': 10,
    'Saturn': 11, 'Nisson': 12, 'BMW': 13, 'Jaguar': 14, 'Porche': 15,
    'Mecedes': 16, 'Ferrari': 17, 'Lexus': 18
}

gender_map = {'Male': 0, 'Female': 1}

maritalstatus_map = {'Married': 0, 'Single': 1, 'Divorced': 2, 'Widow': 3}

fault_map = {'Policy Holder': 0, 'Third Party': 1}

Price_map = {
    'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2,
    '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5
}

policy_type_map = {}
vehicle_category_map = {}


@app.route('/Home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.read_csv('fraud_oracle_cleaned.csv')



    for i, category in enumerate(df['PolicyType'].value_counts().index):
        policy_type_map[category] = i


    for i, category in enumerate(df['VehicleCategory'].value_counts().index):
        vehicle_category_map[category] = i


    days_policy_accident_map = {}
    for i, category in enumerate(df['Days_Policy_Accident'].value_counts().index):
        days_policy_accident_map[category] = i

    days_policy_claim_map = {}
    for i, category in enumerate(df['Days_Policy_Claim'].value_counts().index):
        days_policy_claim_map[category] = i

    past_number_of_claims_map = {}
    for i, category in enumerate(df['PastNumberOfClaims'].value_counts().index):
        past_number_of_claims_map[category] = i

    age_of_vehicle_map = {}
    for i, category in enumerate(df['AgeOfVehicle'].value_counts().index):
        age_of_vehicle_map[category] = i

    police_report_filed_map = {}
    for i, category in enumerate(df['PoliceReportFiled'].value_counts().index):
        police_report_filed_map[category] = i

    witness_present_map = {}
    for i, category in enumerate(df['WitnessPresent'].value_counts().index):
        witness_present_map[category] = i

    agent_type_map = {}
    for i, category in enumerate(df['AgentType'].value_counts().index):
        agent_type_map[category] = i

    number_of_suppliments_map = {}
    for i, category in enumerate(df['NumberOfSuppliments'].value_counts().index):
        number_of_suppliments_map[category] = i

    address_change_claim_map = {}
    for i, category in enumerate(df['AddressChange_Claim'].value_counts().index):
        address_change_claim_map[category] = i

    number_of_cars_map = {}
    for i, category in enumerate(df['NumberOfCars'].value_counts().index):
        number_of_cars_map[category] = i

    base_policy_map = {}
    for i, category in enumerate(df['BasePolicy'].value_counts().index):
        base_policy_map[category] = i

    # Map the input data
    # data['Month'] = map_month[data['Month']]
    data['DayOfWeek'] = map_day[data['DayOfWeek']]
    data['DayOfWeekClaimed'] = map_day[data['DayOfWeekClaimed']]
    data['AccidentArea'] = AccidentArea_map[data['AccidentArea']]
    data['Manufacture_company'] = Manufacture_company_map[data['Manufacture_company']]
    data['Sex'] = gender_map[data['Sex']]
    data['MaritalStatus'] = maritalstatus_map[data['MaritalStatus']]
    data['Fault'] = fault_map[data['Fault']]
    data['MonthClaimed'] = map_month[data['MonthClaimed']]
    data['VehiclePrice'] = Price_map[data['VehiclePrice']]
    data['Days_Policy_Accident'] = days_policy_accident_map[data['Days_Policy_Accident']]
    data['Days_Policy_Claim'] = days_policy_claim_map[data['Days_Policy_Claim']]
    data['PastNumberOfClaims'] = past_number_of_claims_map[data['PastNumberOfClaims']]
    data['AgeOfVehicle'] = age_of_vehicle_map[data['AgeOfVehicle']]
    data['PoliceReportFiled'] = police_report_filed_map[data['PoliceReportFiled']]
    data['WitnessPresent'] = witness_present_map[data['WitnessPresent']]
    data['AgentType'] = agent_type_map[data['AgentType']]
    data['NumberOfSuppliments'] = number_of_suppliments_map[data['NumberOfSuppliments']]
    data['AddressChange_Claim'] = address_change_claim_map[data['AddressChange_Claim']]
    data['NumberOfCars'] = number_of_cars_map[data['NumberOfCars']]
    # data['BasePolicy'] = base_policy_map[data['BasePolicy']]
    data['PolicyType'] = policy_type_map[data['PolicyType']]
    data['VehicleCategory'] = vehicle_category_map[data['VehicleCategory']]

    # Convert data to DataFrame
    df1 = pd.DataFrame([data])

    # Apply scaling
    df_scaled = scaler.fit_transform(df1)
    data_array = df1.to_numpy()
    # Make prediction
    prediction = model.predict(data_array)
    if prediction[0] == 1:
        prediction_text = 'Fraud Found'
    else:
        prediction_text = 'No Fraud Found'

    # return render_template('index.html', prediction_text=prediction_text)

    return render_template('index.html', prediction_text='Predicted Target: {}'.format(prediction_text))


if __name__ == '__main__':
    app.run(debug=True)
