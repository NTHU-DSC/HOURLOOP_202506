import pandas as pd
import pgeocode as pg
import json

fc = pd.read_csv('utils/fc.csv')
fc_code = fc['fc_code']

def check_fc_valid(input_fc: str):
    if input_fc in fc_code.values:
        return True
    else:
        return False
    
def check_postal_valid(input_postal: str):
    if len(input_postal) == 5:
        country = 'us'
        postal = input_postal
    elif len(input_postal) == 10:
        country = 'us'
        postal = input_postal[:5]
    elif len(input_postal) == 4:
        country = 'us'
        postal = "0" + input_postal
    else:
        country = 'ca'
        postal = input_postal

    nomis = {'us': pg.Nominatim('us'), 'ca': pg.Nominatim('ca')}
    geo_info = nomis[country].query_postal_code(postal)
    
    if pd.isna(geo_info['latitude']):
        return False
    else:
        return True
    
def get_past_vendors():
    with open('utils/vendors.json', 'r', encoding='utf-8') as f:
        vendors = json.load(f)
    return vendors

def check_missing_id(id: pd.Series):
    has_missing = id.isnull().any()
    
    if not has_missing:
        return False, id

    missing_mask = id.isnull()
    n_missing = missing_mask.sum()
    
    fill_values = pd.Series(range(n_missing), index=id[missing_mask].index)
    
    filled_id = id.copy()
    filled_id[missing_mask] = fill_values

    return True, filled_id

def check_missing_info(df):
    has_missing = df.isnull().values.any()
    if not has_missing:
        return False, []
    
    missing_positions = [
        (i, col) for i, row in df.iterrows()
        for col in df.columns if pd.isnull(row[col])
    ]
    return True, missing_positions


def check_df_valid(df: pd.DataFrame):
    critical_cols = ['vendor_name', 'fc_code', 'from_postal_code',
                     'total_vendor_price', 'weight', 'volume']
    result = {}

    # 檢查是否有缺少欄位
    missing_cols = [col for col in critical_cols if col not in df.columns]
    if missing_cols:
        result['valid'] = 'missing cols'
        result['return'] = missing_cols
        return result

    # 檢查是否有缺少資料
    has_missing_data, missing_positions = check_missing_info(df[critical_cols])
    if has_missing_data:
        result['valid'] = 'missing data'
        result['return'] = missing_positions
        return result

    # 檢查是否有缺 shipment ID
    if 'Shipment ID' not in df.columns:
        df['Shipment ID'] = range(len(df))
        result['valid'] = 'missing ID'
        result['return'] = df['Shipment ID']
        return result

    has_missing_id, filled_id = check_missing_id(df['Shipment ID'])

    if has_missing_id:
        result['valid'] = 'missing ID'
        result['return'] = filled_id
        return result

    result['valid'] = 'ok'
    result['return'] = None

    return result

# 欄位缺失：valid = 'missing cols' , return = missing_cols
# 資料缺失：valid = 'missing data' , return = missing_positions
# ID 缺失：valid = 'missing ID' , return = new_id
# 都完好：valid = 'ok'