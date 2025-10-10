from flask import Flask, request, render_template
import pandas as pd
import dill  # Make sure you used dill to save your pipeline

app = Flask(__name__)

# Load trained pipeline
with open("pipeline_dill.pkl", "rb") as f:
    pipeline = dill.load(f)

# All numerical and categorical features used in pipeline
numerical_features = [
    'age', 'monthly_income_usd', 'monthly_expenses_usd', 'savings_usd',
    'loan_amount_usd', 'loan_term_months', 'monthly_emi_usd',
    'loan_interest_rate_pct', 'debt_to_income_ratio', 'savings_to_income_ratio'
]

categorical_features = [
    'gender', 'education_level', 'employment_status', 'job_title',
    'has_loan', 'loan_type', 'region'
]

# Define dropdown options for categorical fields
dropdown_options = {
    'gender': ['Male', 'Female', 'Other'],
    'education_level': ['High School', 'Bachelors', 'Masters', 'PhD', 'Other'],
    'employment_status': ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired'],
    'job_title': ['Manager', 'Engineer', 'Technician', 'Clerk', 'Other'],
    'has_loan': ['Yes', 'No'],
    'loan_type': ['Personal', 'Home', 'Auto', 'Education', 'Other'],
    'region': ['North', 'South', 'East', 'West', 'Central', 'Other']
}

all_features = numerical_features + categorical_features

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Collect form data
        data = {}
        for col in numerical_features:
            data[col] = [float(request.form[col])]
        for col in categorical_features:
            data[col] = [request.form[col]]

        df = pd.DataFrame(data, columns=all_features)
        prediction = pipeline.predict(df)[0]
        result = "Yes" if prediction == 1 else "No"

    return render_template('loan.html', result=result, dropdown_options=dropdown_options)

if __name__ == '__main__':
    app.run(debug=True)








