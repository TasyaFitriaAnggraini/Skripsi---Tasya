from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

print(sys.executable)

app = Flask(__name__)

# Load the machine learning model
scaler = joblib.load("./models/scaler_model.joblib")
sfs = joblib.load("./models/sfs_modell.joblib")
model = joblib.load("./models/best_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # Check if the request contains a file
        if 'file' not in request.files:
            print('Debug Info: No file part')
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            print('Debug Info: No selected file')
            return jsonify({'error': 'No selected file'})

        # Read the CSV file
        dataset = pd.read_csv(file)

        # Label Encoder
        le_class = LabelEncoder()
        dataset['class'] = le_class.fit_transform(dataset['class'])
        le_protocol_type = LabelEncoder()
        dataset['protocol_type'] = le_protocol_type.fit_transform(dataset['protocol_type'])
        le_service = LabelEncoder()
        dataset['service'] = le_service.fit_transform(dataset['service'])
        le_flag = LabelEncoder()
        dataset['flag'] = le_flag.fit_transform(dataset['flag'])

        # Create new DataFrame without 'class' column
        dataset_new = dataset.drop(['class'], axis=1)
        
        # Make predictions
        new_data_scaled = scaler.transform(dataset_new)
        new_data_sfs = sfs.transform(new_data_scaled)
        prediction = model.predict(new_data_sfs)

        # Prepare results with predictions
        dataset_new_sfs = pd.DataFrame(new_data_sfs, columns=[dataset_new.columns[i] for i in sfs.k_feature_idx_])
        dataset_new_sfs['class'] = dataset['class']
        dataset_new_sfs['prediksi'] = prediction

        def convert_to_class(prediction):
            return 'anomali' if prediction == 0 else 'normal'
        dataset_new_sfs['prediksi'] = [convert_to_class(pred) for pred in prediction]

        # Inverse transform
        dataset_new_sfs['class'] = le_class.inverse_transform(dataset['class'])
        dataset_new_sfs['protocol_type'] = le_protocol_type.inverse_transform(dataset['protocol_type'])
        dataset_new_sfs['flag'] = le_flag.inverse_transform(dataset['flag'])

        # Convert DataFrame to array for response
        array_data = dataset_new_sfs.to_numpy()
        result_list = [dict(zip(dataset_new_sfs.columns, row)) for row in array_data]

        # Response
        return jsonify(result_list)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
