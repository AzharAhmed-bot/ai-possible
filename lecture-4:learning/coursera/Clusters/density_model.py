import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler
import contextily as ctx
from shapely.geometry import Point

import requests
import zipfile
import io
import os



zip_file_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'
output_dir='./'
os.makedirs(output_dir, exist_ok=True)


response=requests.get(zip_file_url)
response.raise_for_status()


with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):
            zip_ref.extract(file_name,output_dir)
            print(f'Downloaded and Extracted {file_name}')

def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # Show the plot
    plt.savefig('hbdscan_clustered_locations.png')


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df=pd.read_csv(url,encoding="ISO-8859-1")



df=df[['Latitude','Longitude']].apply(pd.to_numeric,errors='coerce')
df=df.dropna()
df[['Latitude','Longitude']]=df[['Latitude','Longitude']].astype('float')

print(df.isnull().sum())

# In this case we know how to scale the coordinates. Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
co_ordinates_scaled=df.copy()
co_ordinates_scaled['Latitude']=2 * co_ordinates_scaled['Latitude']


# model=DBSCAN(eps=1.0,min_samples=3,metric='euclidean')
# df['Cluster']=model.fit_predict(co_ordinates_scaled)
# print(df['Cluster'].value_counts())


# plot_clustered_locations(df,title='Museums Clustered by Proximity')



hdbscan_model=hdbscan.HDBSCAN(min_samples=None,min_cluster_size=3,metric='euclidean')
df['Cluster']=hdbscan_model.fit_predict(co_ordinates_scaled)
print(df['Cluster'].value_counts())
plot_clustered_locations(df,title='Museums Clustered by Proximity')
