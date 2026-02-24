from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))


# Load trained IDS model
model = joblib.load("ids_model.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(1, 42):
        value = float(request.form[f"feature{i}"])
        features.append(value)

    features = np.array([features])

    prediction = model.predict(features)[0]

    result = "NORMAL traffic" if prediction == 0 else "ATTACK detected"

    return render_template("index.html", prediction=result)


# Run server
if __name__ == "__main__":
    app.run(debug=True)
