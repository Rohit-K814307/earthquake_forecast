import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

def interactive_seismic_heatmap(data_dict, output_dir):
    output_file = os.path.join(output_dir, "seismic_heatmap.html")

    flattened_time = []
    flattened_mag = []
    flattened_depth = []
    flattened_lat = []
    flattened_lon = []

    for i in range(len(data_dict["node_id"])):
        node_time = data_dict["time"][i]
        node_mag = data_dict["mag"][i]
        node_depth = data_dict["depth"][i]
        node_lat = data_dict["node_lat"][i]
        node_lon = data_dict["node_lon"][i]
        
        flattened_time.extend(node_time)
        flattened_mag.extend(node_mag)
        flattened_depth.extend(node_depth)
        flattened_lat.extend([node_lat] * len(node_time))  
        flattened_lon.extend([node_lon] * len(node_time))  


    flattened_df = pd.DataFrame({
        "time": flattened_time,
        "mag": flattened_mag,
        "depth": flattened_depth,
        "node_lat": flattened_lat,
        "node_lon": flattened_lon
    })


    fig = px.scatter_mapbox(flattened_df, lat="node_lat", lon="node_lon", color="mag", 
                            size="mag", size_max=15, color_continuous_scale="Viridis", 
                            title="Seismic Events Heatmap", hover_name="time",
                            hover_data=["mag", "depth", "node_lat", "node_lon"])

    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, mapbox_center={"lat": 37.7749, "lon": -122.4194})  
    fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")
    

    def update_nearby_events(trace, points, selector):
        if points.point_inds:
            clicked_lat = flattened_df.iloc[points.point_inds[0]]["node_lat"]
            clicked_lon = flattened_df.iloc[points.point_inds[0]]["node_lon"]


            coords = flattened_df[["node_lat", "node_lon"]].values
            nbrs = NearestNeighbors(n_neighbors=5, radius=0.1)
            nbrs.fit(coords)

            distances, indices = nbrs.kneighbors([[clicked_lat, clicked_lon]])

            nearby_events = flattened_df.iloc[indices[0]]
 
            avg_mag = nearby_events["mag"].mean()
            avg_depth = nearby_events["depth"].mean()

            print("Nearby Events:")
            print(nearby_events[["time", "mag", "depth", "node_lat", "node_lon"]])
            print(f"Average Magnitude: {avg_mag:.2f}")
            print(f"Average Depth: {avg_depth:.2f} km")

            popup_content = f"Average Magnitude: {avg_mag:.2f}<br>Average Depth: {avg_depth:.2f} km"
            
            fig.add_scattermapbox(
                lat=[clicked_lat],
                lon=[clicked_lon],
                mode='markers+text',
                marker=dict(size=12, color="red"),
                text=[popup_content],
                textposition="top right"
            )

    fig.data[0].on_click(update_nearby_events)
    
    fig.write_html(output_file)