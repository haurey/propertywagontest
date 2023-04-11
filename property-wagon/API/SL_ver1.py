import streamlit as st
import pandas as pd
import requests
import numpy as np
from streamlit_folium import st_folium, folium_static
import folium
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title('Property Wagon - HDB resale prices')

# SIDEBAR
st.sidebar.header('Tell us about the HDB you are interested in')
# df_flat_type = pd.DataFrame({'flat_type': ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']})
df_storey_range = pd.DataFrame({'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']})

postal_code = st.sidebar.text_input('Postal Code', 'Enter Postal Code')
# flat_type = st.sidebar.selectbox('Select number of rooms', df_flat_type['flat_type'])
storey_range = st.sidebar.selectbox('Select level of flat', df_storey_range['storey_range'])
submit_button = st.sidebar.button('SUBMIT')

# LOAD DATA
recent_tnx = pd.read_csv('property-wagon/API/data/recent_tnx.csv')

def getcoordinates(postal_code):
    
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    if req.status_code==200:
        resultsdict = eval(req.text)
        if len(resultsdict['results'])>0:
            lat = resultsdict['results'][0]['LATITUDE']
            lon = resultsdict['results'][0]['LONGITUDE']
            blk_no = resultsdict['results'][0]['BLK_NO']
            street_name = resultsdict['results'][0]['ROAD_NAME']
            address = resultsdict['results'][0]['ADDRESS']

            return lat, lon, blk_no, street_name, address
    else:
        st.write('Invalid postal code')
    


def predict(postal_code):
    # get values from postal code
    lat, lon, block, street_name, address = getcoordinates(postal_code)

    #convert long names to short form
    street_short = {
    'AVENUE': 'AVE',
    'BUKIT': 'BT',
    'CENTRAL': 'CTRL',
    'CLOSE': 'CL',
    'COMMONWEALTH': "C'WEATH",
    'CRESCENT': 'CRES',
    'DRIVE': 'DR',
    'HEIGHTS': 'HTS',
    'JALAN': 'JLN',
    'LORONG': 'LOR',
    'NORTH': 'NTH',
    'PARK': 'PK',
    'TANJONG': 'TG',
    'TERRACE': 'TER',
    'PLACE': 'PL',
    'GARDENS': 'GDNS',
    'ROAD': 'RD',
    'SOUTH': 'STH',
    'STREET': 'ST',
    'UPPER': 'UPP',
    }
    for key, val in street_short.items():
        if key in street_name:
            street_name = street_name.replace(key, val)

    # Load data
    train_data = pd.read_csv('property-wagon/API//data/resale-flat-prices-from-2003-2023.csv')
    econ_data = pd.read_csv('property-wagon/API/data/econ_data.csv')

    # Retrieve variables
    town = train_data[train_data['street_name'] == street_name]['town'].head(1).values[0]

    type_model_lease_floor = train_data[(train_data['block'] == block) & (train_data['street_name'] == street_name)][['flat_type', 'flat_model', 'lease_commence_date', 'floor_area_sqm']].drop_duplicates()
    drop_flattype = ['1 ROOM', 'MULTI-GENERATION']
    type_model_lease_floor = type_model_lease_floor[type_model_lease_floor['flat_type'].isin(drop_flattype) == False]

    drop_flatmodel = ['3Gen', '2-room', 'Premium Maisonette',
                  'Improved-Maisonette','Terrace',
                  'Premium Apartment Loft','Type S2',
                  'Type S1','Adjoined flat','Model A-Maisonette',
                  'Multi Generation', 'DBSS', ]
    type_model_lease_floor = type_model_lease_floor[type_model_lease_floor['flat_model'].isin(drop_flatmodel) == False]


    GDP = econ_data['GDP'].head(1).values[0]
    HDB_resale_vol = econ_data['HDB_resale_vol'].head(1).values[0]

    y_pred = []

    for index, row in type_model_lease_floor.iterrows():
        flat_type = row['flat_type']
        flat_model = row['flat_model']
        lease_commence_date = row['lease_commence_date']
        floor_area_sqm = row['floor_area_sqm']

        # query API on Google Cloud Run
        BASE_URI = "https://property-joycetoh-dpqqbevshq-as.a.run.app/predict?"
        query = f'town={town}&flat_type={flat_type}&storey_range={storey_range}&floor_area_sqm={floor_area_sqm}&flat_model={flat_model}&lease_commence_date={lease_commence_date}&GDP={GDP}&HDB_resale_vol={HDB_resale_vol}'
        endpoint = BASE_URI + query
        request = requests.get(endpoint)

        # print(endpoint)

        if request.status_code!=200:
            # st.write(f"Sorry, {query} not found in database.")
            break
        else:
            request=request.json()
            y_pred.append(round(request['HDB Resale Price: $']))

    return y_pred, town

def main():
    
 
    
    if submit_button:
 
        if requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1').status_code!=200 or (requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1').content==b'{"found":0,"totalNumPages":0,"pageNum":1,"results":[]}'): 
            st.write('Invalid postal code, please re-enter.')
            
        else:
            # DISPLAY MAP with RECENT TNX
            lat, lon, blk_no, street_name, address = getcoordinates(postal_code)
            map = folium.Map(location=[lat, lon], zoom_start=16 , control_scale=True)

            # for index, location_info in recent_tnx.iterrows():
            #     folium.CircleMarker(location=[location_info["Latitude"],location_info["Longitude"]],
            #                         radius=5,
            #                         color="crimson",
            #                         fill=True,
            #                         fill_color="crimson",
            #                         popup=location_info[["flat_type", "storey_range", "remaining_lease_yr", "resale_price"]]).add_to(map)

            # ADD PREDICTION
            pred_price,town = predict(postal_code)
            folium.Marker(location=[lat, lon], popup= [postal_code, storey_range, pred_price]).add_to(map)

            # DISPLAY ADDRESS BASED ON POSTAL CODE
            st.write('Address: ', address)

            st_map = folium_static(map, width=800, height=400)
            st.write('Boundaries based on Master Plan 2014 Planning Area Boundary (No Sea)')

            # # WIP : AMENITIES WITHIN 2KM
            # st.header('wip : Nearby Amenities')
            # st.write(recent_tnx.head())
            # # show amenities within 2 km
            # # add column to calculate distance of amentities from address
            # # df_amenities = df_distance[df_distance['distance'] =< 2], sort from smallest distance
            # # st.write(df_amenities)
            
            flattypelist = ['1 ROOM','2 ROOM','3 ROOM','4 ROOM','5 ROOM','EXECUTIVE','MULTI-GENERATION']
            for i in flattypelist:
                pathtofile = Path(f'/app/propertywagontest/property-wagon/propertywagontimeseries/processed_data/{town}{i}.csv')
                if pathtofile.is_file():
                    plot_df = pd.read_csv(pathtofile)
                    fig = px.line(plot_df, x="ds", y="y",line_shape="spline", render_mode="svg")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        # DISPLAY MAP default
        map = folium.Map(location=[1.35, 103.81], zoom_start=12, control_scale=True)
        # choropleth = folium.Choropleth(geo_data='property-wagon/API/data/planning-boundary-area.geojson')
        # choropleth.geojson.add_to(map)
        ### WIP : Adding average price for each planning boundary area

        st_map = folium_static(map, width=800, height=400)
        st.write('Boundaries based on Master Plan 2014 Planning Area Boundary (No Sea)')

        # CREDITS
        st.write('Data Source from data.gov.sg, onemap.sg, and several other online sources')

if __name__ == "__main__":
    main()
