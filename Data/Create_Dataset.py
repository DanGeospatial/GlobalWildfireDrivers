from datetime import datetime, timedelta
import rioxarray
import ee
import dask
import xarray as xr
import xee


def clip_collection(img):
    return img.clip(country_geom)


def get_climate(start: datetime, end: datetime):
    dataset = ee.ImageCollection('UCSB-CHG/CHIRTS/DAILY').filter(ee.Filter.date(
        start, end)).reduce(ee.Reducer.mean())

    minmax_mtmp = dataset.select(['minimum_temperature_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                            geometry=country_geom, scale=100000)
    dataset_mtmp = dataset.select(['minimum_temperature_mean']).unitScale(ee.Number(minmax_mtmp.
                                                                                    get(
        'minimum_temperature_mean_min')), ee.Number(minmax_mtmp.get('minimum_temperature_mean_max')))

    minmax_maxtmp = dataset.select(['maximum_temperature_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                            geometry=country_geom, scale=100000)
    dataset_maxtmp = dataset.select(['maximum_temperature_mean']).unitScale(ee.Number(minmax_maxtmp.
                                                                                    get(
        'maximum_temperature_mean_min')), ee.Number(minmax_maxtmp.get('maximum_temperature_mean_max')))

    minmax_rh= dataset.select(['relative_humidity_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                              geometry=country_geom, scale=100000)
    dataset_rh = dataset.select(['relative_humidity_mean']).unitScale(ee.Number(minmax_rh.
    get(
        'relative_humidity_mean_min')), ee.Number(minmax_rh.get('relative_humidity_mean_max')))

    minmax_hi = dataset.select(['heat_index_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                                 geometry=country_geom, scale=100000)
    dataset_hi = dataset.select(['heat_index_mean']).unitScale(ee.Number(minmax_hi.
    get(
        'heat_index_mean_min')), ee.Number(minmax_hi.get('heat_index_mean_max')))

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    dataset_unitscale = (ee.ImageCollection.fromImages(ee.List([dataset_mtmp, dataset_maxtmp, dataset_rh, dataset_hi]))
                         .map(clip_collection)).map(set_time)

    ds = xr.open_dataset(dataset_unitscale, engine='ee')

    return ds


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


def get_landcover(start: datetime, end: datetime):
    dataset = ee.ImageCollection('MODIS/061/MCD12C1').filter(ee.Filter.date(
        start, end)).reduce(ee.Reducer.mean())

    bands = ee.List(['Land_Cover_Type_3_Percent_Class_0_mean', 'Land_Cover_Type_3_Percent_Class_1_mean',
                     'Land_Cover_Type_3_Percent_Class_2_mean', 'Land_Cover_Type_3_Percent_Class_3_mean',
                     'Land_Cover_Type_3_Percent_Class_4_mean', 'Land_Cover_Type_3_Percent_Class_5_mean',
                     'Land_Cover_Type_3_Percent_Class_6_mean', 'Land_Cover_Type_3_Percent_Class_7_mean',
                     'Land_Cover_Type_3_Percent_Class_8_mean', 'Land_Cover_Type_3_Percent_Class_9_mean',
                     'Land_Cover_Type_3_Percent_Class_10_mean'])

    def process_lc(band):
        lc = dataset.select(band).reproject(crs='EPSG:4326', scale=scale)

        name_min = band + ee.String('_min')
        name_max = band + ee.String('_max')

        minmax_lc = lc.select([band]).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                                geometry=country_geom, scale=100000)
        dataset_lc = lc.select([band]).unitScale(ee.Number(minmax_lc.
        get(
            name_min)), ee.Number(minmax_lc.get(name_max)))

        return dataset_lc

    lc_bands = bands.map(process_lc)

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    dataset_unitscale = (ee.ImageCollection.fromImages(lc_bands)
                         .map(clip_collection)).map(set_time)

    dr_lc = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_lc


def get_topography(start: datetime):
    mtpi_path = ee.Image("CSP/ERGo/1_0/Global/ALOS_mTPI")
    chili_path = ee.Image("CSP/ERGo/1_0/Global/ALOS_CHILI")
    elevation_path = ee.Image("MERIT/DEM/v1_0_3")

    mtpi = mtpi_path.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)
    chili = chili_path.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)
    elevation = elevation_path.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    mtpi_min = mtpi.reduceRegion(reducer=ee.Reducer.min(), scale=100000)
    mtpi_max = mtpi.reduceRegion(reducer=ee.Reducer.max(), scale=100000)
    mtpi_norm = mtpi.unitScale(low=mtpi_min, high=mtpi_max)

    chili_min = chili.reduceRegion(reducer=ee.Reducer.min(), scale=100000)
    chili_max = chili.reduceRegion(reducer=ee.Reducer.max(), scale=100000)
    chili_norm = chili.unitScale(low=chili_min, high=chili_max)

    elevation_min = elevation.reduceRegion(reducer=ee.Reducer.min(), scale=100000)
    elevation_max = elevation.reduceRegion(reducer=ee.Reducer.max(), scale=100000)
    elevation_norm = elevation.unitScale(low=elevation_min, high=elevation_max)

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    dataset_unitscale = (ee.ImageCollection.fromImages(ee.List([mtpi_norm, chili_norm, elevation_norm]))
                         .map(clip_collection)).map(set_time)

    dr_topo = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_topo


def slice_daily(start: datetime, end: datetime):
    merged_xr = xr.merge([get_climate(start=start, end=end), get_wildfire(start=start, end=end),
                          get_landcover(start=start, end=end), get_topography(start=start)])

    return merged_xr


if __name__=='__main__':
    start_time = datetime.fromisoformat('2017-01-01')
    end_time = datetime.fromisoformat('2018-01-01')

    scale = 5566

    ee.Initialize(project='ee-nelson-remote-sensing', url="https://earthengine-highvolume.googleapis.com")
    dask.config.set(scheduler='synchronous', **{'array.slicing.split_large_chunks': True})

    country_geom = ee.FeatureCollection('projects/ee-nelson-remote-sensing/assets/world_simplified').geometry()

    dataset_list = []


    def daily_slices_gen(start: datetime, end: datetime, inclusive_end: bool = False):
        """
        Yield a tuple (day, day+1) for every day in the interval [start, end].

        Parameters
        ----------
        start : datetime
            The first day to include.
        end : datetime
            The last day to consider.  If `inclusive_end` is False, the day
            equal to `end` is *not* yielded; otherwise it is.
        inclusive_end : bool, optional
            If True, the slice that starts on `end` is yielded too.
            (The slice's end will be `end + 1 day`.)
        """
        cur = datetime.combine(start.date(), datetime.min.time(), tzinfo=start.tzinfo)
        limit = datetime.combine(end.date(), datetime.min.time(), tzinfo=end.tzinfo)

        # Move the limit one day forward if we want to include the last day
        if inclusive_end:
            limit += timedelta(days=1)

        while cur < limit:
            # Yield the slice (start, end_of_slice)
            yield cur, cur + timedelta(days=1)
            cur += timedelta(days=1)


    # usage
    for day in daily_slices_gen(start_time, end_time, inclusive_end=False):
        dataset_list.append(slice_daily(start=day[0], end=day[1]))

    dataset_combined = xr.combine_by_coords(dataset_list)
    xr.Dataset.to_zarr(dataset_combined, store="/mnt/d/test.zarr", mode="w-")
