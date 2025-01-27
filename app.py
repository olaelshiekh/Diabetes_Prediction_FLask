from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = [
            float(request.form[field]) for field in [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
        ]
        # Make prediction
        prediction = model.predict([input_data])
        result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': 'Error during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)