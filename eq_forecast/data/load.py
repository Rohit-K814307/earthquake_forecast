import json
import requests
import os
import re
import zipfile

"""
Data Sources:

1. USGS Earthquake API: magnitude, depth, epicenters, timestamp
    - Link: https://earthquake.usgs.gov/fdsnws/event/1/

2. Plate Boundary Lat-Lon Coordinates
    - Link: https://earthquake.usgs.gov/learn/plate-boundaries.kmz

3. Fault Line Lat-Lon Coordinates
    - Link: https://earthquake.usgs.gov/static/lfs/nshm/qfaults/qfaults.kmz
"""


def usgs_earthquake_api(start_time="1950-01-01", end_time="2025-01-01", min_lat=24.396308, max_lat=49.3547868, min_lon=-124.7844079, max_lon=-66.93457, min_magnitude=3.5):

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "minmagnitude": min_magnitude,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        earthquake_data = {
            'time': [],
            'mag': [],
            'lat': [],
            'lon': [],
            'depth': [],
        }

        for feature in data['features']:
            earthquake_data['time'].append(feature['properties']['time'])
            earthquake_data['mag'].append(feature['properties']['mag'])
            earthquake_data['lat'].append(feature['geometry']['coordinates'][1])
            earthquake_data['lon'].append(feature['geometry']['coordinates'][0])
            earthquake_data['depth'].append(feature['geometry']['coordinates'][2])

        return earthquake_data

    else:
        print(f"Failed to fetch data. Status code: {response.status_code}, Error: {response.text}")
        return None

def download_and_extract_kml(kmz_url, min_lat=24.396308, max_lat=49.3547868, min_lon=-124.7844079, max_lon=-66.93457):
    """
    Downloads and extracts KML content from a KMZ file, handling namespaces explicitly.
    """
    response = requests.get(kmz_url)

    if response.status_code == 200:
        with open("eq_forecast/data/raw/data.kmz", 'wb') as f:
            f.write(response.content)

        # Extract KML content from the KMZ
        kmz = zipfile.ZipFile("eq_forecast/data/raw/data.kmz", 'r')
        kml_content = kmz.open('doc.kml', 'r').read().decode('utf-8')  # KML content
        
        coordinates = re.findall(r'<coordinates>(.*?)</coordinates>', kml_content)
        coord_arr = [coord.split(',') for coord in coordinates]

        final_coords = []
        for coord in coord_arr:
            final_coords.append((float(coord[1]), float(coord[0])))

        filtered_coords = []
        for coord in final_coords:
            lat, lon = coord 
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                filtered_coords.append(coord)

        return filtered_coords


    else:
        raise Exception(f"Failed to download KML from {kmz_url}")


def load_agg_data(start_time="1950-01-01", end_time="2025-01-01", min_lat=24.396308, max_lat=49.3547868, min_lon=-124.7844079, max_lon=-66.93457, min_magnitude=3.5, save=False):
    """
    Aggregates earthquake, plate boundary, and fault line data into a single dataset.
    """

    save_path = "eq_forecast/data/raw"
    os.makedirs(save_path, exist_ok=True)
    
    faults_url = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/qfaults.kmz"
    plate_boundaries_url = "https://earthquake.usgs.gov/learn/plate-boundaries.kmz"

    # Download and parse plate boundaries
    print("fetching plate boundaries...")
    plate_boundaries_coords = download_and_extract_kml(kmz_url = plate_boundaries_url,
                                                       min_lat = min_lat,
                                                       max_lat = max_lat,
                                                       min_lon = min_lon,
                                                       max_lon = max_lon
                                                       )
    

    # Download and parse fault lines
    print("fetching fault lines...")
    fault_lines_coords = download_and_extract_kml(kmz_url = faults_url,
                                                    min_lat = min_lat,
                                                    max_lat = max_lat,
                                                    min_lon = min_lon,
                                                    max_lon = max_lon
                                                )

    # Fetch earthquake data
    print("fetching earthquakes...")
    earthquakes = usgs_earthquake_api(start_time=start_time,
                                      end_time=end_time,
                                      min_lat=min_lat,
                                      max_lat=max_lat,
                                      min_lon=min_lon,
                                      max_lon=max_lon,
                                      min_magnitude=min_magnitude)

    # Aggregate data
    data = {
        "plate_boundaries": plate_boundaries_coords, 
        "fault_lines": fault_lines_coords,
        "earthquakes": earthquakes
    }

    # Save to file if requested
    os.remove("eq_forecast/data/raw/data.kmz")
    if save:
        with open(f"{save_path}/eq_data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        os.removedirs("eq_forecast/data/raw")

    return data