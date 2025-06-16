import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import check_data, datapipeline, predict

st.title("Shipping Cost Predictor v1.0")
st.subheader("Hour Loop X NTHU DSC")

tab1, tab2 = st.tabs(["Single Shipment Query", "Batch Upload"])

pipeline = datapipeline.DataPipeline()
model = predict.FinalModel()

with tab1:
    st.session_state['state1'] = 'idle'
    st.subheader("Single Shipment Query")
    with st.form("form for single query"):
        st.write('Please input shipment info')
        col1, col2 = st.columns(2)
        with col1:
            weight_input = st.number_input("Weight", value=1.0, format="%.6f", step=1e-6, min_value = 1e-6, help=None)
            tvp_input = st.number_input("Total Vendor Price", value=1.0, format="%.6f", step=1e-6, min_value = 1e-6)
            volume_input = st.number_input("Volume", value=1.0, format="%.6f", step=1e-6, min_value = 1e-6)

        with col2:
            vendor_name_input = st.selectbox("Vendor Name", check_data.get_past_vendors()+['Others'], help='If your vendor_name does not appear, select "Others" instead.')
            fc_input = st.text_input("FC Code", placeholder='ABC1',help='Need address info in FC database.')
            from_postal_code_input = st.text_input("From Postal Code", placeholder='American/Canadian postal code', help='Valid format: American zip5(XXXXX) or zip9(XXXXX-XXXX), Canadian(XXX XXX)')
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not all([weight_input, tvp_input, volume_input, vendor_name_input, fc_input, from_postal_code_input]):
                st.warning("Please fill in all fields before submitting.")
            elif not check_data.check_fc_valid(fc_input):
                st.error("Invalid FC_code.  Check your spelling or update FC database.")
            elif not check_data.check_postal_valid(from_postal_code_input):
                st.error("Invalid from_postal_code.  See more detail in the hint.")
            else:
                st.success("Shipment info submitted successfully!")
                st.session_state['state1'] = 'got_info'
                st.session_state['time1'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if st.session_state['state1'] == 'got_info':
        st.info(f'Processing data...')
        try:
            st.session_state['input_df1'] = pd.DataFrame({
                'Shipment ID': [f'Query {st.session_state["time1"]}'],
                'vendor_name': [vendor_name_input],
                'fc_code': [fc_input],
                'from_postal_code': [from_postal_code_input],
                'weight': [weight_input],
                'total_vendor_price': [tvp_input],
                'volume': [volume_input]
            })
            #st.dataframe(input_df)
            st.session_state['state1'] = 'ready_for_predicting'
        except:
            st.error('Something went wrong while processing data...')

    if st.session_state['state1'] == 'ready_for_predicting':
        st.info('Predicting shipping cost for each ship method...')
        try:
            df = model.predict(pipeline.process(st.session_state['input_df1']))
            suc = True
        except:
            suc = False
        df = df.drop(columns=['Shipment ID'])
        df.index += 1
        if suc:
            st.success('Successfully predicted!')
            st.session_state['result_df1'] = df
            st.session_state['state1'] = 'show_result'
        else:
            st.error('Something went wrong while predicting cost...')

    if st.session_state['state1'] == 'show_result':
        st.subheader('Predicted Results')
        st.write(f'Query {st.session_state["time1"]}')
        st.dataframe(st.session_state['result_df1'], hide_index = True)
        st.markdown('You may click the **download** button on the top right of dataframe to dowanload the csv file.')
        clear = st.button("Clear")
        if clear:
            st.session_state['state1'] = 'idle'

with tab2:
    st.session_state['state2'] = 'idle'
    st.subheader("Uploading File")
    with st.form("form for batch submit"):
        uploaded_file = st.file_uploader("Please upload an CSV file:", type=["csv"])
        submitted_2 = st.form_submit_button("Submit")
        if submitted_2:
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                    st.session_state['state2'] = 'got_df'
                except:
                    st.error("Uploaded file can not be read.  Check the format.")
            else:
                st.warning("No file uploaded. Please try again.")

    if st.session_state['state2'] == 'got_df':
        st.info("Processing data...")
        d = check_data.check_df_valid(df)
        if d['valid'] == 'missing cols':
            st.error(f"Critical columns {d['return']} missing in uploaded file.  Please check.")
        elif d['valid'] == 'missing data':
            st.error(f"Missing data detected at (row, column) = {d['return']} . Please check.")
        else:
            if d['valid'] == 'missing ID':
                df['Shipment ID'] = d['return']
                st.warning("Some shipment ID missed.  Automatically filled.")
            st.session_state['state2'] = 'check_data'
    
    if st.session_state['state2'] == 'check_data':
        try:
            df['weight'] = pd.to_numeric(df['weight'])
        except:
            st.error('"weight" column contains non-numeric data. Please check.')
            st.session_state['state2'] = 'idle'

        try:
            df['volume'] = pd.to_numeric(df['volume'])
        except:
            st.error('"volume" column contains non-numeric data. Please check.')
            st.session_state['state2'] = 'idle'

        try:
            df['total_vendor_price'] = pd.to_numeric(df['total_vendor_price'])
        except:
            st.error('"total_vendor_price" column contains non-numeric data. Please check.')
            st.session_state['state2'] = 'idle'

        df['from_postal_code'] = df['from_postal_code'].astype(str)
        
        valid = 1

        bar = st.progress(0)
        hint = st.empty()
        progress = 0

        for i in range(len(df)):
            progress = (i + 1)/len(df)
            hint.write(f'{np.round(progress*100, 1)}% | Checking shipment ID {df["Shipment ID"].iloc[i]} ...')
            if not check_data.check_fc_valid(df.iloc[i]['fc_code']):
                st.error(f"Invalid fc_code at shipment ID {df['Shipment ID'].iloc[i]} .  Please check.")
                valid = 0
            if not check_data.check_postal_valid(df.iloc[i]['from_postal_code']):
                st.error(f"Invalid from_postal_code at shipment ID {df['Shipment ID'].iloc[i]} .  Please check.")
                valid = 0
            bar.progress(progress)
            

        if valid:
            st.success("Uploaded data accepted!")
            st.session_state['input_df2'] = df[['Shipment ID', 'weight', 'volume', 'total_vendor_price',
                                                'vendor_name', 'fc_code', 'from_postal_code']]
            st.session_state['state2'] = 'ready_for_predicting'

        else:
            st.session_state['state2'] = 'idle'

    if st.session_state['state2'] == 'ready_for_predicting':
        st.info('Predicting shipping cost for each ship method...')
        try:
            df = model.predict(pipeline.process(st.session_state['input_df2']))
            suc = True
        except:
            suc = False
        
        if suc:
            st.success('Successfully predicted!')
            st.session_state['result_df2'] = df
            st.session_state['state2'] = 'show_result'
        else:
            st.error('Something went wrong while predicting cost...')

    if st.session_state['state2'] == 'show_result':
        st.subheader('Predicted Results')
        st.dataframe(st.session_state['result_df2'], hide_index = True)
        st.markdown('You may click the **download** button on the top right of dataframe to dowanload the csv file.')
        clear = st.button("Clear")
        if clear:
            st.session_state['state2'] = 'idle'
