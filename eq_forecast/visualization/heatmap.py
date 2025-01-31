import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import plotly.express as px
import plotly.graph_objects as go

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

####################Make prediction heatmaps##################
def create_time_slider_map(df, feature_name):
    """
    Creates a Plotly map with a time slider from a given dataframe.
    
    Parameters:
        df (pd.DataFrame): A dataframe containing 'time', 'lat', 'lon', and a feature column.
        feature_name (str): The name of the feature column to visualize.
    
    Returns:
        fig (plotly.graph_objects.Figure): The generated figure with a time slider.
    """
    df = df.sort_values(by='time')
    times = df['time'].unique()
    
    fig = go.Figure()
    
    # Create frames for each time step
    frames = []
    for time in times:
        df_time = df[df['time'] == time]
        frames.append(
            go.Frame(
                data=[go.Scattermapbox(
                    lat=df_time['lat'],
                    lon=df_time['lon'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=df_time[feature_name],
                        colorscale='Viridis',
                        showscale=True,
                        cmin=0,
                        cmax=6
                    ),
                    text=df_time[feature_name]
                )],
                name=str(time)
            )
        )
    
    # Add initial data
    df_init = df[df['time'] == times[0]]
    fig.add_trace(
        go.Scattermapbox(
            lat=df_init['lat'],
            lon=df_init['lon'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_init[feature_name],
                colorscale='Viridis',
                showscale=True
            ),
            text=df_init[feature_name]
        )
    )
    
    # Define slider steps
    steps = []
    for i, time in enumerate(times):
        step = dict(
            method='animate',
            args=[[str(time)],
                  dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=str(time)
        )
        steps.append(step)
    
    # Create layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
            zoom=3
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])
            ]
        )],
        sliders=[dict(
            steps=steps,
            active=0,
            x=0.1, 
            y=0,
            len=0.9
        )],
        title=f'Time Evolution of {feature_name}'
    )
    
    fig.frames = frames
    return fig


def make_pred_heatmaps(preds, actuals, grid, output_dir_pred, output_dir_act, feature_name, feature_focus=0):
    #input shape of actuals & preds: (batch_size, num_nodes, num_features, num_timesteps)
    preds = preds[:,:,feature_focus,:].squeeze(2).numpy() #new shape: (batch_size, num_nodes, num_timesteps)
    actuals = actuals[:,:,feature_focus,:].squeeze(2).numpy() #new shape: (batch_size, num_nodes, num_timesteps)

    #average across all batches
    preds = np.mean(preds, axis=0) #new shape: (num_nodes, num_time_steps)
    actuals = np.mean(actuals, axis=0) #new shape: (num_nodes, num_time_steps)

    #get location dfs
    df_preds = {"node":[i for i in range(len(preds))], 
          "lat":[c[0] for c in grid["centers"]], 
          "lon":[c[1] for c in grid["centers"]], 
          "timesteps":preds}
    
    df_actuals = {"node":[i for i in range(len(actuals))], 
          "lat":[c[0] for c in grid["centers"]], 
          "lon":[c[1] for c in grid["centers"]], 
          "timesteps":actuals}

    nodes_pred = []
    lats_pred = []
    lons_pred = []
    times_pred = []
    feature_pred = []

    nodes_act = []
    lats_act = []
    lons_act = []
    times_act = []
    feature_act = []

    for i in range(len(df_preds["timesteps"])):
        timestep_pred = df_preds["timesteps"][i]
        timestep_act = df_actuals["timesteps"][i]
        lat = df_preds["lat"][i]
        lon = df_preds["lon"][i]
        node = i

        for step in range(len(timestep_pred)):
            times_pred.append(step)
            times_act.append(step)
            lats_pred.append(lat)
            lats_act.append(lat)
            lons_pred.append(lon)
            lons_act.append(lon)
            nodes_pred.append(node)
            nodes_act.append(node)
            feature_pred.append(timestep_pred[step])
            feature_act.append(timestep_act[step])

    
    df_act = pd.DataFrame({"time":times_act, "lat":lats_act, "lon":lons_act, feature_name:feature_act})
    df_pred = pd.DataFrame({"time":times_pred, "lat":lats_pred, "lon":lons_pred, feature_name:feature_pred})


    fig_act = create_time_slider_map(df_act, feature_name)
    fig_pred = create_time_slider_map(df_pred, feature_name)

    fig_act.write_html(output_dir_act)
    fig_pred.write_html(output_dir_pred)



    




    


        

        



        



    
    
    







