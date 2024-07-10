from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset and pre-trained model
data = pd.read_csv('final_dataset.csv')
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

@app.route('/')
def index():
    # Retrieve unique values for dropdown menus
    bedrooms_list = sorted(data['beds'].unique())
    bathrooms_list = sorted(data['baths'].unique())
    sizes_list = sorted(data['size'].unique())
    zip_codes_list = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms_list=bedrooms_list, 
                           bathrooms_list=bathrooms_list, sizes_list=sizes_list, 
                           zip_codes_list=zip_codes_list, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zip_code = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zip_code]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    # Convert numeric columns to appropriate types
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    # Predict the price
    prediction = model.predict(input_data)[0]

    # Render the same template with the prediction result
    return render_template('index.html', bedrooms_list=sorted(data['beds'].unique()), 
                           bathrooms_list=sorted(data['baths'].unique()), sizes_list=sorted(data['size'].unique()), 
                           zip_codes_list=sorted(data['zip_code'].unique()), prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
