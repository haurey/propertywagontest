import streamlit as st
import pandas as pd
import requests
import numpy as np
from streamlit_folium import st_folium, folium_static
import folium

def main():
    st.set_page_config(layout="wide")
    st.title('Property Wagon - HDB resale prices')

    # DISPLAY MAP
    map = folium.Map(location=[1.35, 103.81], zoom_start=12, control_scale=True)
    choropleth = folium.Choropleth(geo_data='data/planning-boundary-area.geojson')
    choropleth.geojson.add_to(map)
    ### WIP : Adding average price for each planning boundary area
    st_map = st_folium(map, width=1400, height=700)
    st.write('Boundaries based on Master Plan 2014 Planning Area Boundary (No Sea)')

    # LOAD DATA
    recent_tnx = pd.read_csv('data/recent_tnx.csv')


    # CREDITS
    st.write('Data Source from data.gov.sg, onemap.sg, and several other online sources')

    # SIDEBAR
    st.sidebar.header('Tell us about the HDB you are interested in')
    df_flat_type = pd.DataFrame({'flat_type': ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']})
    df_storey_range = pd.DataFrame({'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']})

    address = st.sidebar.text_input('Address', 'Enter Address')
    flat_type = st.sidebar.selectbox('Select number of rooms', df_flat_type['flat_type'])
    storey_range = st.sidebar.selectbox('Select level of flat', df_storey_range['storey_range'])
    submit_button = st.sidebar.button('SUBMIT')

    if submit_button:
        # ADD RECENT TNX
        address_lat = getcoordinates(address)[0]
        address_long = getcoordinates(address)[1]

        map2 = folium.Map(location=[address_lat, address_long], zoom_start=17, control_scale=True)

        for index, location_info in recent_tnx.iterrows():
            folium.CircleMarker(location=[location_info["Latitude"],location_info["Longitude"]],
                                radius=5,
                                color="crimson",
                                fill=True,
                                fill_color="crimson",
                                popup=location_info[["flat_type", "storey_range", "remaining_lease_yr", "resale_price"]]).add_to(map2)

        # ADD PREDICTION
        pred_price = predict(address)
        folium.Marker(location=[address_lat, address_long], popup= [address, pred_price]).add_to(map2)

        st_map = st_folium(map2, width=1400, height=700)
        st.write('Boundaries based on Master Plan 2014 Planning Area Boundary (No Sea)')

        # WIP : AMENITIES WITHIN 2KM
        st.header('wip : Nearby Amenities')
        st.write(recent_tnx.head())
        # show amenities within 2 km
        # add column to calculate distance of amentities from address
        # df_amenities = df_distance[df_distance['distance'] =< 2], sort from smallest distance
        # st.write(df_amenities)

def getcoordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass

def predict(address):
    # ENTER THE API FOR PREDICTED PRICE
    pred_px=600000
    return pred_px

if __name__ == "__main__":
    main()
