from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
lr_model=pickle.load(open('salmodel.pkl','rb'))
col_list=pickle.load(open('column_list.pkl','rb'))

@app.route('/')

def model():
    return 'welcome to sal pred model'

@app.route('/salary_pred')

def salary_pred():
    data=request.get_json()
    Age=data['Age']
    Weight=data['Weight']
    Experience=data['Experience']

    data_frame={"Age":[Age],"Weight":[Weight],"Experience":[Experience]}

    test_data=pd.DataFrame(data_frame)

    salary=lr_model.predict(test_data)

    return jsonify ({"salary":salary[0]})

if __name__== "__main__":
    app.run(host='0.0.0.0',port=5012)
