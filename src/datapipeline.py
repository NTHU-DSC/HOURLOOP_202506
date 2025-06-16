import pickle
import pandas as pd
import numpy as np
import pgeocode as pg


class DataPipeline:
    def __init__(self):
        self.ship_methods = ["WWE_PARCEL", "WWE_LTL", "UBER_LTL", "ESTES", "HOUR_LOOP_FEDEX_PARCEL", "AMAZON_UPS_PARCEL", "AMAZON_LTL", "AMAZON_FREIGHT"]
        
        self.te_features = ["from_state", "to_state", "vendor_name"]
        self.target_encoders = self._load_encoders()

    def _load_encoders(self):
        encoders = {feat: {} for feat in self.te_features}
        
        for feature_to_encode in self.te_features:
            for ship_method in self.ship_methods:
                with open(f'encoders/{feature_to_encode}/{feature_to_encode}_{ship_method}.pkl', 'rb') as f:
                    encoders[feature_to_encode][ship_method] = pickle.load(f)
        return encoders

    def _get_geo_info(self, data: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame({
            "from_latitude": np.zeros(len(data)),
            "from_longitude": np.zeros(len(data)),
            "from_city": np.zeros(len(data)),
            "from_state": np.zeros(len(data))
        })
        
        # Handle with different formats of zip codes
        zips = []
        for i, zip in enumerate(data):
            zip = str(zip)
            if len(zip) == 5:
                zips.append(('us', i, zip))
            elif len(zip) == 10:
                zips.append(('us', i, zip[:5]))
            elif len(zip) == 4:
                zips.append(('us', i, "0"+zip))
            else:
                zips.append(('ca', i, zip))

        # Get geo info of from adress
        nomis = {'us': pg.Nominatim('us'), 'ca': pg.Nominatim('ca')}
        for (country, i, zip) in zips:
            geo_info = nomis[country].query_postal_code(zip)
            result_df['from_latitude'].iloc[i] = geo_info['latitude']
            result_df['from_longitude'].iloc[i] = geo_info['longitude']
            result_df['from_city'].iloc[i] = geo_info['place_name']
            result_df['from_state'].iloc[i] = geo_info['state_code']

        return result_df

    def _get_distence(self, data: pd.DataFrame) -> tuple[pd.DataFrame]:
        R = 6371
        lat1, lat2 = np.radians(data['from_latitude']), np.radians(data['to_latitude'])
        lon1, lon2 = np.radians(data['from_longitude']), np.radians(data['to_longitude'])
        dlat, dlon = (lat2 - lat1), (lon2 - lon1)
        
        # Calculate Haversine distance
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        log_Hdis = np.log1p(R*c).astype(np.float32)

        # Calculate Manhattan distance
        dlat_dist = R * np.abs(dlat)
        avg_lat = (lat1 + lat2) / 2
        dlon_dist = R * np.abs(dlon) * np.cos(avg_lat)
        log_Mdis = np.log1p(dlat_dist+dlon_dist).astype(np.float32)

        return log_Hdis, log_Mdis
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:

        # Initialize output dataframe with Shipment ID, fc_code and vendor_name
        output_df = pd.DataFrame(data[["Shipment ID", "fc_code", "vendor_name"]])

        # Log transform weight, volume and TVP
        output_df["log_weight"] = np.log1p(data["weight"]).astype(np.float32)
        output_df["log_volume"] = np.log1p(data["volume"]).astype(np.float32)
        output_df["log_TVP"] = np.log1p(data["total_vendor_price"]).astype(np.float32)

        # Handle with from adress
        from_geo_info = self._get_geo_info(data["from_postal_code"])
        output_df = pd.concat([output_df, from_geo_info], axis=1)

        # Handle with FC geo info (to adress)
        fc = pd.read_csv("utils/fc.csv")
        output_df = pd.merge(output_df, fc, on='fc_code', how='left')

        # Get Haversine and Manhattan distance
        log_Hdis, log_Mdis = self._get_distence(output_df)
        output_df["log_Hdis"], output_df["log_Mdis"] = log_Hdis, log_Mdis

        # Extra Interactive features
        output_df['log_TVP/log_weight'] = output_df['log_TVP'] / output_df['log_weight']
        output_df['across_state'] = (output_df['from_state'] != output_df['to_state']).astype(int)

        # Drop columns useless for prediction
        output_df = output_df.drop(["from_latitude", "to_latitude", "from_longitude", "to_longitude", "Unnamed: 0"], axis=1)

        # Target encoding
        for feature_to_encode in self.te_features:
            for ship_method in self.ship_methods:
                output_df[f"{feature_to_encode}_encoded_{ship_method}"] = output_df[feature_to_encode].map(self.target_encoders[feature_to_encode][ship_method]['map']).fillna(self.target_encoders[feature_to_encode][ship_method]['fallback'])
        
        return output_df


# Functional Testing
if __name__ == "__main__":
    sample = pd.DataFrame({
        "Shipment ID": ["0001", "0002", "0003", "0004", "0005"],
        "vendor_name": ["Two's Company", "R&R Corp.", "Glimmer Goddess", "Pet Lou", "Glimmer Goddess"],
        "fc_code": ["ICT2", "TMB8", "TMB8", "SCK8", "TMB8"],
        "from_postal_code": ["10523", "20852", "6795", "14151-0888", "L2E 0A6"],
        "total_vendor_price": [1.2, 3.5, 4.6, 2.36, 23.5],
        "weight": [123., 456., 789., 1211., 659.],
        "volume": [1000., 2000., 3.5, 4., 502.6]
    })
    pipeline = DataPipeline()
    processed_sample = pipeline.process(sample)
    processed_sample.to_csv("ohyeah.csv")
    print(processed_sample)
    