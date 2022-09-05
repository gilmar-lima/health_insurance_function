from flask import Response
from models import HealthInsurance

import functions_framework
import joblib

@functions_framework.http
def predict(request):

    load_data()

    try:
        content_type = request.headers.get('Content-Type')
        method = request.method

        if(method != 'POST'): 
            return Response( '{"message" : "HTTP method not supported"}', 
                                status=400, 
                                mimetype='application/json'
                            )

        if(content_type != 'application/json'): 
            return Response( '{"message" : "Content-Type not supported"}', 
                                status=400, 
                                mimetype='application/json'
                            )

        health_insurance = HealthInsurance(_model,_column_transformer,
                                            _bins_annual_premium_type)

        payload = request.data.decode('utf-8')
        payload_predicted = health_insurance.predict(payload)

        return Response( payload_predicted, 
                        status=200, 
                        mimetype='application/json')
        
    except Exception as e:
        return Response( '{}'.format(str(e)), 
                        status=400, 
                        mimetype='application/json')

def load_data():
    global _model 
    global _column_transformer
    global _bins_annual_premium_type

    _model = joblib.load(filename = 'parameters/random_forrest.gz')
    _column_transformer = joblib.load(filename = 'parameters/column_transformer.joblib')
    _bins_annual_premium_type = joblib.load(filename = 'parameters/bins_annual_premium_type.joblib')