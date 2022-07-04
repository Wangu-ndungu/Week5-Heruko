# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:14:06 2022

@author: user
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle


app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0])
           
    return render_template('index.html', prediction_text= 'Predicted final exam score ${}'.format(output))
       
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)    
    
      
           
        
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=os.environ.get('PORT', '5000'))
