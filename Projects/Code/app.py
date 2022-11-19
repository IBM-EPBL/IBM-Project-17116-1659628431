

import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename

from flask import Flask, redirect, render_template, request, url_for

app=Flask(__name__)

model = load_model("vegetable.h5")
model1=load_model("fruit.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        f=request.files.get('image','')
        basepath=os.path.dirname(__file__)
       
        file_path=os.path.join(
            basepath,'',secure_filename(f.filename))
    
        f.save(file_path)
        img=image.load_img(file_path,target_size=(128,128))
     
        x=image.img_to_array(img)
       
        x=np.expand_dims(x,axis=0)
        print(x)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            preds=model.predict_classes(x)
            print(preds)
            print(preds[0])
            df=pd.read_excel(r'../precautions -vegetabels.xlsx')
            print(df)
            print(df.iloc[preds[0]]['caution'])
        else:
            preds=model1.predict_classes(x)
            print(preds)

            print(preds[0])
            df=pd.read_excel(r'../precautions -vegetabels.xlsx')
            print(df)
            print(df.iloc[preds[0]]['caution'])
        # return df.iloc[preds[0]]['caution']
        # return preds[0]
        # print(request.form['plant'])
        return df.iloc[preds[0]]['caution']
    
    return "hello"

if __name__ == "__main__":
    app.run(debug=False)