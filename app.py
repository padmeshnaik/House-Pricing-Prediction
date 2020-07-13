import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,url_for
import pickle
from sklearn.preprocessing import PolynomialFeatures




app=Flask(__name__)
model= pickle.load(open("C:/Users/Padmesh/Desktop/Machine Learning Models/Regression Models/House_Prices/model.pkl","rb"))


@app.route('/',methods=['POST','GET'])
def home():
    return render_template('Home.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method=='POST':
        bathrooms=request.form['BR']
        Area_of_Living=float(request.form['AOL'])
        No_of_views=request.form['NOV']
        Grade=request.form['G']
        Area_of_basement=float(request.form['AOB'])
        lat=request.form['L']
        

        Area_of_Living_logged=np.log(Area_of_Living)
        Area_of_basement_logged=np.log1p(Area_of_basement)

        int_features=[]
        int_features.append(bathrooms)
        int_features.append(Area_of_Living_logged)
        int_features.append(No_of_views)
        int_features.append(Grade)
        int_features.append(Area_of_basement_logged)
        int_features.append(lat)

        final_features = [np.array(int_features)]
        X_poly=PolynomialFeatures(degree=2)
        X_features=X_poly.fit_transform(final_features)
        ans = model.predict(X_features)

        final_ans=np.exp(ans)
        for x in final_ans:
            final_ans1=float(x)
        formatted_ans = "{:.2f}".format(final_ans1)

    return render_template('predict.html', prediction=formatted_ans)

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Another':
            return redirect("/")
        