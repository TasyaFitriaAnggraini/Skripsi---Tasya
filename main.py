from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

print(sys.executable)

app = Flask(__name__)

# Load the machine learning model
scaler = joblib.load("./models/scaler_search.joblib")
sfs = joblib.load("./models/sfs_model.joblib")
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
        # 'class'
        le_class = LabelEncoder()
        dataset['class'] = le_class.fit_transform(dataset['class'])
        # 'protocol_type'
        le_protocol_type = LabelEncoder()
        dataset['protocol_type'] = le_protocol_type.fit_transform(dataset['protocol_type'])
        # 'service'
        le_service = LabelEncoder()
        dataset['service'] = le_service.fit_transform(dataset['service'])
        # 'flag'
        le_flag = LabelEncoder()
        dataset['flag'] = le_flag.fit_transform(dataset['flag'])

        # Buat DataFrame baru tanpa kolom 'class'
        dataset_new = dataset.drop(['class'], axis=1)
        
        # Make predictions
        new_data_scaled = scaler.transform(dataset_new)
        new_data_sfs = sfs.transform(new_data_scaled)
        prediction = model.predict(new_data_sfs)

        # Misalnya, new_data_rfe adalah hasil transformasi menggunakan RFE pada data yang telah discaler
        dataset_new_sfs = pd.DataFrame(new_data_sfs, columns=dataset_new.columns[sfs.support_])
        # Menambahkan kolom 'class' asli ke dalam DataFrame dataset_new_rfe
        dataset_new_sfs['class'] = dataset_new['class']
        # Menambahkan hasil prediksi ke dalam DataFrame dataset_new_rfe
        dataset_new_sfs['prediksi'] = prediction

        # Membuat fungsi untuk konversi nilai prediksi menjadi string
        def convert_to_class(prediction):
            return 'anomali' if prediction == 0 else 'normal'
        dataset_new_sfs['prediksi'] = [convert_to_class(pred) for pred in prediction]

        # Mengembalikan nilai ke bentuk awal
        dataset_new_sfs['class'] = le_class.inverse_transform(dataset['class'])
        dataset_new_sfs['protocol_type'] = le_protocol_type.inverse_transform(dataset['protocol_type'])
        dataset_new_sfs['flag'] = le_flag.inverse_transform(dataset['flag'])

        # Mengonversi DataFrame ke dalam bentuk array
        array_data = dataset_new_sfs.to_numpy()

        # List dengan nama kolom
        result_list = [dict(zip(dataset_new_sfs.columns, row)) for row in array_data]

        # Response
        return jsonify(result_list)

    except Exception as e:
        print('Debug Info: Exception -', str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
