import pickle
import numpy as np
from flask import Flask, jsonify, request, render_template

app= Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route("/")

def Home():
  return render_template("file.html")


@app.route("/predict", methods = ["POST"])

def predict():
  float_features=[float(x) for x in request.form.values()]
  features = [np.array(float_features)]
  prediction = model.predict(features)
  return render_template("file.html", prediction_text = "The species name of iris flower is {} ".format(prediction))

  
if __name__=="__main__":
  app.run(debug=True)
