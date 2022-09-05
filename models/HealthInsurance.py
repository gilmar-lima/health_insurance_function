
import pandas as pd


class HealthInsurance():

    def __init__(self, model, column_transformer, bins_annual_premium_type):
        """model : the sklearn model already trainned.
           colums_transformer : the column transformer with all transformations.
           bins_annual_premium_type : bins to create annual_premium_type feature"""

        self.model = model
        self.transformer = column_transformer
        self.bins_annual_premium_type = bins_annual_premium_type


    def feature_engineering(self, df):
        premium_categories = ['very_low', 'low', 'moderate', 'high', 'very_high']

        df['vehicle_age'] = df['vehicle_age'].apply(self.get_vehicle_age)
        df['annual_premium_type'] = pd.cut(x = df['annual_premium'], 
                                            bins = self.bins_annual_premium_type,
                                            labels = premium_categories)
        return df

    def get_vehicle_age(self, vehicle_age):

        vehicle_labels = { 
            '> 2 Years' : 'over_2_years',
            '1-2 Year' : 'between_1_2_year',
            '< 1 Year' : 'below_1_year'
            }
        
        return vehicle_labels.get(vehicle_age)

    def data_preparation(self, df):        
        return self.transformer.transform(df)
    
    def predict(self, payload):
        df = pd.read_json(payload, orient='records')

        np_array = (df.pipe(self.feature_engineering)
                         .pipe(self.data_preparation)
                    )

        df['score'] = self.model.predict_proba(np_array)[:, 1]

        return df.to_json(orient='records')



    






