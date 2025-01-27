# Diabetes Prediction App
![image](https://github.com/user-attachments/assets/ab25dff7-1711-45a0-affa-81250e2d2bb3)

This is a simple Flask-based web application for predicting diabetes using the Pima Indians Diabetes Database. The app allows users to input health data and predicts whether they have diabetes or not.

## Features
- User-friendly form for inputting health data.
- Predicts whether a person has diabetes or not.
- Automatically reloads the page after displaying the prediction.

## Dataset
The dataset used for this project is the **Pima Indians Diabetes Database** from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). It contains health-related features such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## How to Run

### Prerequisites
- Python 3.x
- Flask
- Scikit-learn
- Joblib

### Installation
1. Clone this repository:
   ```bash
   https://github.com/olaelshiekh/Diabetes_Prediction_FLask.git
   ```

2.Navigate to the project folder:
  ```bash 
  cd diabetes-prediction-app
  ```

3.Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the App
1. Start the Flask app:
```bash
python app.py
```

2. Open your browser and go to ``*http://127.0.0.1:5000/*``

### Using the App
1.Fill in the form with the required health data.

2.Click Predict to see the result.

3.The page will automatically reload after 5 seconds to reset the form.


# Project Structure
```bash
diabetes-prediction-app/
│
├── app.py                  # Flask application
├── model_pipeline.pkl      # Trained machine learning model
├── templates/              # HTML templates
│   └── index.html          # Frontend form
├── README.md               # Project documentation
```




 
