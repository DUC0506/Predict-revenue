from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import numpy as np
import joblib
import pandas as pd
import lightgbm as lgb
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
MODEL_PATH = './main.pkl'
model = joblib.load(MODEL_PATH)

MODEL_FILE='./lgb_model.pkl'
loaded_model = joblib.load(MODEL_FILE)
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_sales():
    json_data = request.json
    start_date = json_data['start_date']
    end_date = json_data['end_date']
    print(start_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    # Convert JSON data to DataFrame
    # input_data = pd.DataFrame.from_dict(json_data, orient='index').T

    input_data = pd.DataFrame(index=date_range)
    input_data['Store'] = 1
    input_data['Dept'] = 1
    input_data['IsHoliday'] = 0
    input_data['Type'] = 2
    input_data['Size'] = 151315
    input_data['Temperature'] = 75.55
    input_data['Fuel_Price'] = 3.749
    input_data['MarkDown1'] = 21290.130859
    input_data['MarkDown2'] = 0.0
    input_data['MarkDown3'] = 0
    input_data['MarkDown4'] = 0
    input_data['MarkDown5'] = 0
    input_data['CPI'] = 221.7
    input_data['Unemployment'] = 7.0
    input_data['Day'] = input_data.index.day
    input_data['Week'] = input_data.index.isocalendar().week
    input_data['Month'] = input_data.index.month
    input_data['Quarter'] = input_data.index.quarter
    input_data['Year'] = input_data.index.year
    
    # Kiểm tra và cập nhật các ngày là ngày lễ
    holidays = json_data.get('holidays', [])  # Danh sách ngày lễ từ JSON
    for holiday in holidays:
        holiday_date = pd.to_datetime(holiday)
        if holiday_date in input_data.index:
            input_data.loc[holiday_date, 'IsHoliday'] = 1
    # Perform prediction
    prediction = model.predict(input_data)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})
def create_sample_data(start_date, end_date, store_id, item_id):
    date_range = pd.date_range(start=start_date, end=end_date ,freq='W')
    sample_data = pd.DataFrame(columns=[
        'store', 'item', 'day_of_month', 'day_of_year', 'week_of_year', 
        'is_wknd', 'is_month_start', 'is_month_end', 
        'sales_lag_91', 'sales_lag_98', 'sales_lag_105', 'sales_lag_112', 
        'sales_lag_119', 'sales_lag_126', 'sales_lag_182', 'sales_lag_364', 
        'sales_lag_546', 'sales_lag_728', 'sales_roll_mean_365', 
        'sales_roll_mean_546', 'sales_roll_mean_730', 
        'sales_ewm_alpha_099_lag_91', 'sales_ewm_alpha_099_lag_98', 
        'sales_ewm_alpha_099_lag_105', 'sales_ewm_alpha_099_lag_112', 
        'sales_ewm_alpha_099_lag_180', 'sales_ewm_alpha_099_lag_270', 
        'sales_ewm_alpha_099_lag_365', 'sales_ewm_alpha_099_lag_546', 
        'sales_ewm_alpha_099_lag_728', 'sales_ewm_alpha_095_lag_91', 
        'sales_ewm_alpha_095_lag_98', 'sales_ewm_alpha_095_lag_105', 
        'sales_ewm_alpha_095_lag_112', 'sales_ewm_alpha_095_lag_180', 
        'sales_ewm_alpha_095_lag_270', 'sales_ewm_alpha_095_lag_365', 
        'sales_ewm_alpha_095_lag_546', 'sales_ewm_alpha_095_lag_728', 
        'sales_ewm_alpha_09_lag_91', 'sales_ewm_alpha_09_lag_98', 
        'sales_ewm_alpha_09_lag_105', 'sales_ewm_alpha_09_lag_112', 
        'sales_ewm_alpha_09_lag_180', 'sales_ewm_alpha_09_lag_270', 
        'sales_ewm_alpha_09_lag_365', 'sales_ewm_alpha_09_lag_546', 
        'sales_ewm_alpha_09_lag_728', 'sales_ewm_alpha_08_lag_91', 
        'sales_ewm_alpha_08_lag_98', 'sales_ewm_alpha_08_lag_105', 
        'sales_ewm_alpha_08_lag_112', 'sales_ewm_alpha_08_lag_180', 
        'sales_ewm_alpha_08_lag_270', 'sales_ewm_alpha_08_lag_365', 
        'sales_ewm_alpha_08_lag_546', 'sales_ewm_alpha_08_lag_728', 
        'sales_ewm_alpha_07_lag_91', 'sales_ewm_alpha_07_lag_98', 
        'sales_ewm_alpha_07_lag_105', 'sales_ewm_alpha_07_lag_112', 
        'sales_ewm_alpha_07_lag_180', 'sales_ewm_alpha_07_lag_270', 
        'sales_ewm_alpha_07_lag_365', 'sales_ewm_alpha_07_lag_546', 
        'sales_ewm_alpha_07_lag_728', 'sales_ewm_alpha_05_lag_91', 
        'sales_ewm_alpha_05_lag_98', 'sales_ewm_alpha_05_lag_105', 
        'sales_ewm_alpha_05_lag_112', 'sales_ewm_alpha_05_lag_180', 
        'sales_ewm_alpha_05_lag_270', 'sales_ewm_alpha_05_lag_365', 
        'sales_ewm_alpha_05_lag_546', 'sales_ewm_alpha_05_lag_728', 
        'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 
        'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'month_1', 
        'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 
        'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
    ])
    sample_data['store'] = store_id
    sample_data['item'] = item_id
    sample_data['day_of_month'] = np.random.randint(1, 28, len(date_range))
    sample_data['day_of_year'] = np.random.randint(1, 365, len(date_range))
    sample_data['week_of_year'] = np.random.randint(1, 52, len(date_range))
    sample_data['is_wknd'] = np.random.choice([True, False], size=len(date_range))
    sample_data['is_month_start'] = np.random.choice([True, False], size=len(date_range))
    sample_data['is_month_end'] = np.random.choice([True, False], size=len(date_range))
    
    # Randomly fill sales-related columns
    for col in sample_data.columns[7:70]:
        sample_data[col] = np.random.randint(1, 1000, len(date_range))
    
    # Randomly fill day_of_week and month columns
    for col in sample_data.columns[70:]:
        sample_data[col] = np.random.choice([0, 1], size=len(date_range))
    
    return sample_data
@app.route('/predict-item/<int:item_id>', methods=['POST'])
def predict_item(item_id):
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    store_id = 4
    sample_data = create_sample_data(start_date, end_date, store_id, item_id)
    predictions = loaded_model.predict(sample_data)
    return jsonify({'predictions': predictions.tolist()})
if __name__ == '__main__':
    app.run(debug=True)