from flask import Flask, request

import pickle
import pandas as pd
import json


app = Flask(__name__)
# app = Flask(__name__, template_folder='/home/jackim/mysite/templates/')

# load the model from disk 
filename = 'finalized_model.sav' # Absolute path is needed for deployment at python anywhere    
# filename = '/home/jackim/mysite/static/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route("/")
def hello_world():
    return "<p>TEAM2 ML Model API</p>"

@app.route("/prediction1", methods = ['GET', 'POST'])
def prediction1() :    
    # read the json from the request
    json_data = request.json

    # json -> dic
    a_json = json.loads(json_data)

    # dic -> dataframe
    df = pd.DataFrame.from_dict(a_json)

    X_instance = df[["Primary_Offence", "diff", "Premises_Type", "Cost_of_Bike", "Occurrence_DayOfWeek", "Occurrence_Year", "Report_Hour", "Occurrence_Month", "Location_Type", "NeighbourhoodName", "Longitude", "Latitude"]]
    y_instance = df[["Status"]]

    prediction = loaded_model.predict(X_instance)        

    # prediction and y_instance are not number but nd_array(?) so that can't be jsonified
    # even the values are taken out, the types are still int32, which can't be jsonified too! So convert into normal int
    dictionary = { "prediction" : int(prediction[0]), "actual_value" : int(y_instance.values[0][0])}    
    json_string = json.dumps(dictionary, indent = 4)
    
    
    return json_string

@app.route("/prediction2", methods = ['GET', 'POST'])
def prediction2() :    
    # read the json from the request
    json_data = request.json

    # json -> dic
    a_json = json.loads(json_data)

    # dic -> dataframe
    df = pd.DataFrame.from_dict(a_json)

    X_instance = df[["Primary_Offence", "diff", "Premises_Type", "Cost_of_Bike", "Occurrence_DayOfWeek", "Occurrence_Year", "Report_Hour", "Occurrence_Month", "Location_Type", "NeighbourhoodName", "Longitude", "Latitude"]]    

    prediction = loaded_model.predict(X_instance)        

    # prediction and y_instance are not number but nd_array(?) so that can't be jsonified
    # even the values are taken out, the types are still int32, which can't be jsonified too! So convert into normal int
    dictionary = { "prediction" : int(prediction[0])}    
    json_string = json.dumps(dictionary, indent = 4)   
    
    return json_string

@app.route("/run", methods = ['GET', 'POST'])
def run():
    # load the model from disk 
    # filename = 'finalized_model.sav' # Absolute path is needed for deployment at python anywhere    
    # loaded_model = pickle.load(open(filename, 'rb'))

    # read the json from the request
    json_data = request.json

    # json -> dic
    a_json = json.loads(json_data)

    df = pd.DataFrame.from_dict(a_json)
    
    
    X_test = df[["Primary_Offence", "diff", "Premises_Type", "Cost_of_Bike", "Occurrence_DayOfWeek", "Occurrence_Year", "Report_Hour", "Occurrence_Month", "Location_Type", "NeighbourhoodName", "Longitude", "Latitude"]]
    y_test = df[["Status"]]

    # send the result as json again
    result = loaded_model.score(X_test, y_test)        

    dictionary = { "Model Score" : result}
    
    json_object = json.dumps(dictionary, indent = 4)
    
    
    return json_object
    

if __name__ ==  "__main__":
    app.run(debug=True)


