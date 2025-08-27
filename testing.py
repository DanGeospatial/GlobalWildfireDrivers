from datetime import datetime
import ee
import xarray as xr

start_time = datetime.fromisoformat('2017-08-01')
end_time = datetime.fromisoformat('2017-08-02')
start_time_b = datetime.fromisoformat('2017-08-02')
end_time_b = datetime.fromisoformat('2017-08-03')
scale = 5566

ee.Initialize(project='ee-nelson-remote-sensing', url="https://earthengine-highvolume.googleapis.com")
"""
country_geom = ee.FeatureCollection('projects/ee-nelson-remote-sensing/assets/world_simplified').geometry()
dataset_list = []


def clip_collection(img):
    return img.clip(country_geom)


def get_wildfire(start: datetime, end: datetime):
    dataset = ee.ImageCollection("FIRMS").filter(ee.Filter.date(
        start, end)).reduce(ee.Reducer.mean()).select('T21_mean')

    fires = dataset.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_fires = fires.select(['T21_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                           geometry=country_geom, scale=100000)
    dataset_fires = fires.select(['T21_mean']).unitScale(ee.Number(minmax_fires.
    get(
        'T21_mean_min')), ee.Number(minmax_fires.get('T21_mean_max')))

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    dataset_unitscale = (ee.ImageCollection.fromImages(ee.List([dataset_fires]))
                         .map(clip_collection)).map(set_time)

    dr_fires = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_fires


dataset_list.append(get_wildfire(start_time, end_time))
dataset_list.append(get_wildfire(start_time_b, end_time_b))

comb = xr.combine_by_coords(dataset_list)

import matplotlib.pyplot as plt

comb.T21_mean.isel(time=1).plot()
plt.show()
"""
if ee.Algorithms.If((ee.Number(0).gt(ee.Number(0.1))), 'True', 'False').getInfo() == 'True':
    print("Here")
else:
    print("Bottom")