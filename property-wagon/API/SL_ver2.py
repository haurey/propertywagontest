import streamlit as st
import pandas as pd
import requests
import numpy as np
from streamlit_folium import st_folium, folium_static
import folium
from IPython.display import display
import pickle
from pathlib import Path
import plotly.express as px
import base64


print("path is") 
print(Path.cwd())
st.set_page_config(layout="wide")

st.title('Property Wagon - HDB Resale Price Predictor')

# SIDEBAR
st.sidebar.header('Enter the postal code and floor level to find out more!')
# df_flat_type = pd.DataFrame({'flat_type': ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']})
df_storey_range = pd.DataFrame({'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48']})

postal_code = st.sidebar.text_input('Postal Code', 'Enter Postal Code')
# flat_type = st.sidebar.selectbox('Select number of rooms', df_flat_type['flat_type'])
storey_range_test = st.sidebar.selectbox('Floor level', df_storey_range['storey_range'])
submit_button = st.sidebar.button('SUBMIT')


def getcoordinates(postal_code):
    req = requests.get('https://www.onemap.gov.sg/api/common/elastic/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        lat = resultsdict['results'][0]['LATITUDE']
        lon = resultsdict['results'][0]['LONGITUDE']
        blk_no = resultsdict['results'][0]['BLK_NO']
        street_name = resultsdict['results'][0]['ROAD_NAME']
        address = resultsdict['results'][0]['BLK_NO']+" "+resultsdict['results'][0]['ROAD_NAME']

        return lat, lon, blk_no, street_name, address
    else:
        return None


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
    train_data = pd.read_csv('/mount/src/propertywagontest/property-wagon/API/data/resale-flat-prices-from-2003-2023.csv')
    econ_data = pd.read_csv('/mount/src/propertywagontest/property-wagon/API/data/econ_data.csv')

    # Retrieve variables
    town_test = train_data[train_data['street_name'] == street_name]['town'].head(1).values[0]

    type_model_lease_floor = train_data[(train_data['block'] == block) & (train_data['street_name'] == street_name)][['flat_type', 'flat_model', 'lease_commence_date', 'floor_area_sqm']].drop_duplicates()
    drop_flattype = ['1 ROOM', 'MULTI-GENERATION']
    type_model_lease_floor = type_model_lease_floor[type_model_lease_floor['flat_type'].isin(drop_flattype) == False]

    drop_flatmodel = ['3Gen', '2-room', 'Premium Maisonette',
                  'Improved-Maisonette','Terrace',
                  'Premium Apartment Loft','Type S2',
                  'Type S1','Adjoined flat','Model A-Maisonette',
                  'Multi Generation', 'DBSS', ]
    type_model_lease_floor = type_model_lease_floor[type_model_lease_floor['flat_model'].isin(drop_flatmodel) == False]


    GDP_test = econ_data['GDP'].head(1).values[0]
    HDB_resale_vol_test = econ_data['HDB_resale_vol'].head(1).values[0]

    y_pred = []

    # load model
    with open('/mount/src/propertywagontest/property-wagon/API/data/model', 'rb') as m:
        model = pickle.load(m)
    # load pipeline
    with open('/mount/src/propertywagontest/property-wagon/API/data/columntransformer', 'rb') as c:
        columntransformer = pickle.load(c)

    for index, row in type_model_lease_floor.iterrows():
        flat_type_test = row['flat_type']
        flat_model_test = row['flat_model']
        lease_commence_date_test = row['lease_commence_date']
        floor_area_sqm_test = row['floor_area_sqm']

        # query API on Google Cloud Run
        # BASE_URI = "https://property-joycetoh-dpqqbevshq-as.a.run.app/predict?"
        # query = f'town={town}&flat_type={flat_type}&storey_range={storey_range}&floor_area_sqm={floor_area_sqm}&flat_model={flat_model}&lease_commence_date={lease_commence_date}&GDP={GDP}&HDB_resale_vol={HDB_resale_vol}'
        # endpoint = BASE_URI + query
        # request = requests.get(endpoint).json()

        # if not request:
        #     print(f"Sorry, {query} not found in database.")
        #     return None
        # else:
        #     y_pred.append(round(request['HDB Resale Price: $']))

        X_test = pd.DataFrame(dict(
           town=[town_test],
           flat_model=[flat_model_test],
           flat_type=[flat_type_test],
           storey_range=[storey_range_test],
           floor_area_sqm=[floor_area_sqm_test],
           lease_commence_date=[lease_commence_date_test],
           GDP=[GDP_test],
           HDB_resale_vol=[HDB_resale_vol_test]
       ))

        X_test_processed = columntransformer.transform(X_test)
        pred = model.predict(X_test_processed)
        y_pred.append(int(pred))


    y_pred_df = pd.DataFrame (y_pred, columns = ['Predicted Price'])
    popup_df = pd.concat([type_model_lease_floor.reset_index(), y_pred_df.reset_index()], axis=1)
    popup_df.drop(columns='index', inplace=True)
    popup_df.sort_values(by=['flat_type', 'floor_area_sqm'], ascending=True, inplace=True)
    popup_df.rename(columns={"flat_type": "Flat Type",
                             "flat_model": "Flat Model",
                             "lease_commence_date": "Lease Commcence Date",
                             "floor_area_sqm": "Floor Area (sqm)"},  inplace=True)
    popup_df.index = np.arange(1,len(popup_df)+1)
    popup_df['Predicted Price'] = popup_df['Predicted Price'].astype('str')
    popup_df['Predicted Price'] =  popup_df['Predicted Price'].apply(lambda x : f'${x[0:-3]},{x[-3:]}' if len(x)<=6 else f'${x[0]},{x[1:4]},{x[-3:]}' )
    # return type_model_lease_floor
    # return y_pred_df
    return popup_df, town_test

def main():
    # if getcoordinates(postal_code) is None:
    #     return st.write('Invalid postal code, please enter a valid postal code.')

    if submit_button:
        if requests.get('https://www.onemap.gov.sg/api/common/elastic/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1').status_code!=200 or (requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=Y&pageNum=1').content==b'{"found":0,"totalNumPages":0,"pageNum":1,"results":[]}'): 
            st.write('Invalid postal code, please re-enter.')
        else: 


            lat, lon, blk_no, street_name, address = getcoordinates(postal_code)

            # DISPLAY ADDRESS BASED ON POSTAL CODE
            st.write('Address: ', address)

            # Display map
            map = folium.Map(location=[lat, lon], zoom_start=17, control_scale=True)
            tooltip = "Click for more details!"

            # Price prediction
            popup_df, town = predict(postal_code)
            popup_html = popup_df.to_html(
                classes="table table-striped table-hover table-condensed table-responsive"
                )

            # Popup marker
            popup = folium.Popup(popup_html, max_width=500)
            folium.Marker(location=[lat, lon], popup=popup,  tooltip=tooltip).add_to(map)


            # folium.Marker([30, -100], popup=popup).add_to(m)
            st_map = folium_static(map, width=800, height=400)
            
            flattypelist = ['1 ROOM','2 ROOM','3 ROOM','4 ROOM','5 ROOM','EXECUTIVE','MULTI-GENERATION']
            for i in flattypelist:
                pathtofile = Path(f'/mount/src/propertywagontest/property-wagon/propertywagontimeseries/processed_data/{town}{i}.csv')
                if pathtofile.is_file():
                    plot_df = pd.read_csv(pathtofile)
                    plot_df.rename(columns={'y':'Resale_Price','ds':'Date'},inplace=True)
                    plot_df['Date'] = plot_df['Date'].astype('datetime64')
                    plot_df1 = plot_df[plot_df['Date']>=pd.to_datetime("2023-03-01")]
                    plot_df1.rename(columns={'Resale_Price':'Forecasted_Price($)'},inplace=True)
                    plot_df2 = plot_df[plot_df['Date']<=pd.to_datetime("2023-03-01")]
                    plot_df2.rename(columns={'Resale_Price':'Price($)'},inplace=True)
                    plot_df3 = pd.merge(plot_df2,plot_df1,how='outer',on='Date')
                    fig = px.line(plot_df3, x="Date", y=["Price($)","Forecasted_Price($)"],line_shape="spline", render_mode="svg",title=f'Average Resale {i} HDB Price in {town}')
                    fig.update_layout(yaxis_title="Price($)",xaxis_title="Year",
                        title = {
                                        'y':0.9, # new
                                        'x':0.4,
                                        'xanchor': 'center',
                                        'yanchor': 'top' # new
                                        })
                    st.plotly_chart(fig, use_container_width=True)
            st.write('Boundaries are based on "Master Plan 2014 Planning Area Boundary"')
            st.write('Data sources: Data.gov.sg, Onemap.sg & other online sources.')

    

    else:
        # DISPLAY MAP default
        
        map = folium.Map(location=[1.361516, 103.803701], zoom_start=11, control_scale=True)
        medium_px = pd.read_csv('/mount/src/propertywagontest/property-wagon/API/data/hdb_median_prices_by_town.csv')
        choropleth = folium.Choropleth(geo_data='/mount/src/propertywagontest/property-wagon/API/data/merged_gdf.geojson',
                               data=medium_px,
                               columns=('Name','4-ROOM'),
                               key_on='feature.properties.Name',fill_color="Reds",
                               fill_opacity=1.0)
        # Display Town Label
        choropleth.geojson.add_to(map)
        choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields=["Name","4-ROOM"],aliases=["Town:","Price($):"] ,labels=True))

        st.write('Hover cursor over the map to view median price of HDB in each town.(As of Mar 2023)')
        
        folium_static(map, width=950, height=550)
        
        
        # CREDITS
        st.write('Boundaries are based on "Master Plan 2014 Planning Area Boundary"')
        st.write('Data sources: Data.gov.sg, Onemap.sg & other online sources.')       
        
    

if __name__ == "__main__":
    main()
