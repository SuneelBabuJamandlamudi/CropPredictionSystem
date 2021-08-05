from flask import Flask, request,render_template
import pickle

import warnings
warnings.simplefilter("ignore")

import numpy as np
app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/ff',methods = ['POST', 'GET'])
def ff():
   return render_template("predict.html")

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
   values=[]
   
   crop=request.form['name']
   values.append(crop)
   
   pH=request.form['pH']
   values.append(pH)
   
   temp=request.form['temp']
   values.append(temp)
   
   rainfall=request.form['rainfall']
   values.append(rainfall)
   
   humidity=request.form['humidity']
   values.append(humidity)
   
   final_values=[np.array(values)]
   
   prediction=model.predict(final_values)
   
   result=prediction

   if(result==0):
      textmsg="Crop to be grown is Wheat"
   elif(result==1):
      textmsg="Crop to be grown is Rice"         
   elif(result==2):
      textmsg="Crop to be grown is Maize"                
   elif(result==3):
      textmsg="Crop to be grown is Green gram"
   elif(result==4):
      textmsg= "Crop to be grown is Pea"
   elif(result==5):
      textmsg="Crop to be grown is pigeon pea"
   elif(result==6):
      textmsg="Crop to be grown is Sunflower"  
   elif(result==7):
      textmsg="Crop to be grown is Onion"
   elif(result==8):
      textmsg="Crop to be grown is Millets"
   elif(result==9):
      textmsg="Crop to be grown is Potato"
   elif(result==10):
      textmsg="Crop to be grown is Sugarcane"
   elif(result==11):
      textmsg="Crop to be grown  is Cotton"
   elif(result==12):
      textmsg="Crop to be grown is Soyabean"
   
   if result==0:
       return render_template('result.html',rrr=textmsg)
   else:
       return render_template('result.html',rrr=textmsg)


if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
