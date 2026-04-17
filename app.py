from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('/template/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "age": int(request.form['age']),
            "sex": request.form['sex'],
            "bmi": float(request.form['bmi']),
            "children": int(request.form['children']),
            "smoker": request.form['smoker'],
            "region": request.form['region']
        }

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return render_template(
            'template/index.html',
            prediction_text=f"Estimated Cost: ₹{round(prediction, 2)}"
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)