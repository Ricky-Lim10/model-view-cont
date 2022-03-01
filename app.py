from flask import Flask,jsonify,request
from code import getprediction
app = Flask(__name__)
@app.route("/predict-digit",methods=["POST"])
def predictdigit():
    image=request.files.get("digit")
    prediction=getprediction(image)
    return jsonify({
        "prediction":prediction
    }),200
if __name__ =="__main__":
    app.run(debug=True)