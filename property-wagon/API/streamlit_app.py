import streamlit as st
import pandas as pd
import requests
import numpy as np
from streamlit_folium import folium_static
import folium
import os
import pandas as pd

st.set_page_config(layout="wide")
st.title('Property Wagon - HDB resale prices')





# let taxiFareApiUrl = 'https://taxifare.lewagon.ai/predict'; // replace with your API endpoint
# const centralCoordinates = [1.290270, 103.851959]; // starting position [lng, lat]

# if (window.location.href.includes('https://taxifare.lewagon.com')) {
#   taxiFareApiUrl = 'https://taxifare.lewagon.ai/predict';}
# mapboxgl.accessToken = 'pk.eyJ1Ijoia3Jva3JvYiIsImEiOiJja2YzcmcyNDkwNXVpMnRtZGwxb2MzNWtvIn0.69leM_6Roh26Ju7Lqb2pwQ';

# const displayMap = (start, stop) => {
#   const map = new mapboxgl.Map({
#     container: 'map',
#     style: 'mapbox://styles/mapbox/streets-v11', // stylesheet location
#     center: centralCoordinates,
#     zoom: 10 // starting zoom});




st.sidebar.header('Tell us about the HDB you are interested in')

df_flat_type = pd.DataFrame({'flat_type': ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']})
df_storey_range = pd.DataFrame({'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']})

address = st.sidebar.text_input('Address', 'Enter Address')
flat_type = st.sidebar.selectbox('Select number of rooms', df_flat_type['flat_type'])
storey_range = st.sidebar.selectbox('Select level of flat', df_storey_range['storey_range'])
submit_button = st.sidebar.button('SUBMIT')

st.sidebar.header('Nearby Amenities')

def geo(address):
    response = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    result = response.json()

    return (train, mall, hawker, school)

if submit_button:
    
#         # Make API call and display results in main section
#         response = requests.get(f"https://some-api.com?address={address}&flat_type={flat_type}&storey_range={storey_range}")
#         result = response.json()
    
    train = 'api nearest train and distance'
    mall = 'api nearest mall and distance'
    hawker = 'api nearest hawker and distance'
    school = 'api nearest school and distance'
    supermarket = 'api nearest supermarket and distance'
    st.sidebar.write(train)
    st.sidebar.write(mall)
    st.sidebar.write(hawker)
    st.sidebar.write(school)
                     
st.subheader('Add a map') 
def get_map_data():
    return pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
df = get_map_data()
st.map(df)
  
    
st.subheader('Try another map')     
m = folium.Map(location=[1.290270, 103.85195], zoom_start=6)

geojson_path = os.path.join("getAllPlanningarea.json")
resale_path = os.path.join("recent_tnx.csv")

for _, month in pd.read_csv(resale_path).iterrows():

    folium.Marker(
        location=[month.Latitude, month.Longitude],
        popup=[month.resale_price, month.address, month.month],
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    
