from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model (2).pkl','rb'))

@app.route('/')
def home():
    return render_template('Nasopharyngeal.html')

@app.route('/predict', methods=['POST'])
def predict():
    age=int(request.values['age'])
    gender=int(request.values['gender'])
    air_polution_input=int(request.values['air_polution_input'])
    alcohol_use_input=int(request.values['alcohol_use_input'])
    smoking_input=int(request.values['smoking_input'])
    passive_smoker_input=int(request.values['passive_smoker_input'])
    frequent_cold_input=int(request.values['frequent_cold_input'])
    dry_cough_input=int(request.values['dry_cough_input'])
    snoring_input=int(request.values['snoring_input'])
     

    features=([age,gender,air_polution_input,alcohol_use_input,smoking_input,passive_smoker_input,frequent_cold_input,dry_cough_input,snoring_input])
    print(features)
    new_features=np.array(features).reshape(1,9)
    prediction=model.predict(new_features)
    prediction_values={0:"HIGH",1:"LOW",2:"MEDIUM"}
    result=prediction_values[prediction[0]]
    return render_template ('Nasopharyngeal.html', prediction="Your Nasopharyngeal Cancer is in {} level".format(result))



if __name__=="__main__":
    app.run(port=8000)