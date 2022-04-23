from flask import Flask, request

import pickle
import pandas as pd
import json

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>TEAM2 ML Model API</p>"

@app.route("/run", methods = ['POST'])
def run():
    # load the model from disk 
    filename = 'finalized_model.sav' # Absolute path is needed for deployment at python anywhere    
    loaded_model = pickle.load(open(filename, 'rb'))

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


