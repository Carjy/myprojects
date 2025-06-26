from flask import Flask, request, render_template
import numpy as np
import pickle
import warnings
from feature import FeatureExtraction

# Ignorer les warnings
warnings.filterwarnings('ignore')

# Charger le modèle entraîné
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

# Initialiser l'application Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        # Extraire les caractéristiques de l'URL
        features = FeatureExtraction(url).getFeaturesList()
        x = np.array(features).reshape(1, 30)

        # Prédiction
        y_pro_safe = gbc.predict_proba(x)[0, 1]  # proba que le site soit safe
        return render_template("index.html", xx=round(y_pro_safe, 2), url=url)

    # GET method : aucune prédiction
    return render_template("index.html", xx=-1, url="")

if __name__ == "__main__":
    app.run(debug=True)
