from flask import Flask, request
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)


#MAC
#FLASK_APP=server.py flask run

#WINDOWS
#$env:FLASK_APP = 'server.py'; flask run

#booster
bst = lgb.Booster(model_file='./archive/final.txt');

#clusters
crime_clusters = pickle.load( open( "./archive/Crime_Cluster.pkl", "rb" ) );
total_clusters = pickle.load( open( "./archive/Total_Clusters.pkl", "rb" ) );

#label encoders
body_style = pickle.load( open( "./archive/Body_Style_encoder.pkl", "rb" ) );
color = pickle.load( open( "./archive/Color_encoder.pkl", "rb" ) );
issue_date = pickle.load( open( "./archive/Issue_Date_encoder.pkl", "rb" ) );
location = pickle.load( open( "./archive/Location_encoder.pkl", "rb" ) );
meter_id = pickle.load( open( "./archive/Meter_Id_encoder.pkl", "rb" ) );
route = pickle.load( open( "./archive/Route_encoder.pkl", "rb" ) );
rp = pickle.load( open( "./archive/RP_encoder.pkl", "rb" ) );
vin = pickle.load( open( "./archive/VIN_encoder.pkl", "rb" ) );
violation = pickle.load( open( "./archive/Violation_Description_encoder.pkl", "rb" ) );
code = pickle.load( open("./archive/Violation_code_encoder.pkl", "rb") );

#category encoder
category = pickle.load( open( "./archive/Cat_encoder.pkl", "rb" ) );


@app.route("/", methods=["POST"])
def index():
    if request.method == 'POST':
        json_pred = request.get_json();
        data = pd.DataFrame(json_pred, index=[0]);
        data['Body Style'] = body_style.get(data['Body Style'][0], 1e9);
        data['Color'] = color.get(data['Color'][0], 1e9);
        data['Issue Date'] = issue_date.get(data['Issue Date'][0], 1e9);
        data['Location'] = location.get(data['Location'][0], 1e9);
        data['Meter Id'] = meter_id.get(data['Meter Id'][0], 1e9);
        data['Route'] = route.get(data['Route'][0], 1e9);
        data['RP State Plate'] = rp.get(data['RP State Plate'][0], 1e9);
        data['VIN'] = vin.get(data['VIN'][0], 1e9);
        data['Violation Description'] = violation.get(data['Violation Description'][0], 1e9);
        data['Violation code'] = code.get(data["Violation code"][0], 1e9);
        data = data.drop(['Ticket number'], axis=1);
        
        clusters = crime_clusters.predict(data[["Latitude", "Longitude"]])
        data["crime_clusters"] = clusters

        clusters2 = total_clusters.predict(data)
        data["total_clusters"] = clusters

        data["Issue Time"] = data["Issue Time"].apply(lambda x: float(x))
        data["Marked Time"] = data["Marked Time"].apply(lambda x: float(x))
        data["Plate Expiry Date"] = data["Plate Expiry Date"].apply(lambda x: float(x))
        data["Agency"] = data["Agency"].apply(lambda x: float(x))
        data["Fine amount"] = data["Fine amount"].apply(lambda x: float(x)) 
        data["Latitude"] = data["Latitude"].apply(lambda x: float(x))
        data["Longitude"] = data["Longitude"].apply(lambda x: float(x))       
        data = category.transform(data);
        
        preds = bst.predict(data);
        return np.array2string(preds);