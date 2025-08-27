from datetime import datetime, timedelta
import ee
import dask
import xarray as xr
import xee


def clip_collection(img):
    return img.clip(country_geom)


def get_climate(start: datetime, end: datetime):
    dataset = ee.ImageCollection('UCSB-CHG/CHIRTS/DAILY').filter(ee.Filter.date(
        start.isoformat(), end.isoformat())).reduce(ee.Reducer.mean())

    maxtmp_clamp = dataset.select(['maximum_temperature_mean']).clamp(-89, 60)
    mintmp_clamp = dataset.select(['minimum_temperature_mean']).clamp(-89, 60)

    minmax_mtmp = mintmp_clamp.select(['minimum_temperature_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                            geometry=country_geom, scale=100000)
    dataset_mtmp = mintmp_clamp.select(['minimum_temperature_mean']).unitScale(ee.Number(minmax_mtmp.
                                                                                    get(
        'minimum_temperature_mean_min')), ee.Number(minmax_mtmp.get('minimum_temperature_mean_max')))

    minmax_maxtmp = maxtmp_clamp.select(['maximum_temperature_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                            geometry=country_geom, scale=100000)
    dataset_maxtmp = maxtmp_clamp.select(['maximum_temperature_mean']).unitScale(ee.Number(minmax_maxtmp.
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

    stacked = ee.Image(dataset_mtmp).addBands(dataset_maxtmp).addBands(dataset_rh).addBands(dataset_hi)
    dataset_unitscale = (ee.ImageCollection([stacked])
                         .map(clip_collection)).map(set_time)

    ds = xr.open_dataset(dataset_unitscale, engine='ee')

    return ds


def get_wildfire(start: datetime, end: datetime):
    dataset = ee.ImageCollection("FIRMS").filter(ee.Filter.date(
        start.isoformat(), end.isoformat()))
    dataset_mean = dataset.reduce(ee.Reducer.mean()).select('T21_mean')
    fires = dataset_mean.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_fires = fires.select(['T21_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                           geometry=country_geom, scale=100000)

    if minmax_fires.get('T21_mean_min').getInfo() is None:
        dataset_fires = ee.Image.constant(0).rename('T21_mean').reproject(crs='EPSG:4326', scale=scale)
    else:
        dataset_fires = fires.select(['T21_mean']).unitScale(ee.Number(minmax_fires.
        get('T21_mean_min')).subtract(0.1), ee.Number(minmax_fires.get('T21_mean_max')))


    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    dataset_unitscale = (ee.ImageCollection([dataset_fires])
                         .map(clip_collection)).map(set_time)

    dr_fires = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_fires


def get_landcover(start: datetime):
    start_lc = str(start.year) + '-' + str('01') + '-' + str('01')
    end_lc = str(start.year) + '-' + str('12') + '-' + str('31')

    dataset = ee.ImageCollection('MODIS/061/MCD12C1').filter(ee.Filter.date(
        start_lc, end_lc)).reduce(ee.Reducer.mean())

    bands = ee.List(['Land_Cover_Type_3_Percent_Class_0_mean', 'Land_Cover_Type_3_Percent_Class_1_mean',
                     'Land_Cover_Type_3_Percent_Class_2_mean', 'Land_Cover_Type_3_Percent_Class_3_mean',
                     'Land_Cover_Type_3_Percent_Class_4_mean', 'Land_Cover_Type_3_Percent_Class_5_mean',
                     'Land_Cover_Type_3_Percent_Class_6_mean', 'Land_Cover_Type_3_Percent_Class_7_mean',
                     'Land_Cover_Type_3_Percent_Class_8_mean', 'Land_Cover_Type_3_Percent_Class_9_mean',
                     'Land_Cover_Type_3_Percent_Class_10_mean'])

    def process_lc(band):
        lc = dataset.select([band]).reproject(crs='EPSG:4326', scale=scale)

        name_min = ee.String(band).cat(ee.String('_min'))
        name_max = ee.String(band).cat(ee.String('_max'))

        minmax_lc = lc.select([band]).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                                geometry=country_geom, scale=100000)
        dataset_lc = lc.select([band]).unitScale(ee.Number(minmax_lc.
        get(
            name_min)), ee.Number(minmax_lc.get(name_max)))

        return dataset_lc

    comp = bands.map(process_lc)

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    stacked = (ee.Image(comp.get(0)).addBands(comp.get(1)).addBands(comp.get(2)).addBands(comp.get(3))
               .addBands(comp.get(4)).addBands(comp.get(5)).addBands(comp.get(6)).addBands(comp.get(7))
               .addBands(comp.get(8)).addBands(comp.get(9)).addBands(comp.get(10)))

    dataset_unitscale = (ee.ImageCollection([stacked])
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

    mtpi_min = mtpi.reduceRegion(reducer=ee.Reducer.min(), geometry=country_geom, scale=100000)
    mtpi_max = mtpi.reduceRegion(reducer=ee.Reducer.max(), geometry=country_geom, scale=100000)
    mtpi_norm = mtpi.unitScale(low=ee.Number(mtpi_min.get('AVE')), high=ee.Number(mtpi_max.get('AVE')))

    chili_min = chili.reduceRegion(reducer=ee.Reducer.min(), geometry=country_geom, scale=100000)
    chili_max = chili.reduceRegion(reducer=ee.Reducer.max(), geometry=country_geom, scale=100000)
    chili_norm = chili.unitScale(low=ee.Number(chili_min.get('constant')), high=ee.Number(chili_max.get('constant')))

    elevation_min = elevation.reduceRegion(reducer=ee.Reducer.min(), geometry=country_geom, scale=100000)
    elevation_max = elevation.reduceRegion(reducer=ee.Reducer.max(), geometry=country_geom, scale=100000)
    elevation_norm = elevation.unitScale(low=ee.Number(elevation_min.get('dem')), high=ee.Number(elevation_max.get('dem')))

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    stacked = ee.Image(mtpi_norm).addBands(chili_norm).addBands(elevation_norm)
    dataset_unitscale = (ee.ImageCollection([stacked])
                         .map(clip_collection)).map(set_time)

    dr_topo = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_topo


def get_ecology(start: datetime, end: datetime):
    dataset_evi = ee.ImageCollection("MODIS/061/MOD13A2").filter(ee.Filter.date(
        start.isoformat(), end.isoformat())).reduce(ee.Reducer.mean()).select('EVI_mean')

    dataset_et = ee.ImageCollection("CAS/IGSNRR/PML/V2_v018").filter(ee.Filter.date(
        start.isoformat(), end.isoformat())).reduce(ee.Reducer.mean()).select('Ec_mean')

    dataset_gpp = ee.ImageCollection("CAS/IGSNRR/PML/V2_v018").filter(ee.Filter.date(
        start.isoformat(), end.isoformat())).reduce(ee.Reducer.mean()).select('GPP_mean')

    dataset_es = ee.ImageCollection("CAS/IGSNRR/PML/V2_v018").filter(ee.Filter.date(
        start.isoformat(), end.isoformat())).reduce(ee.Reducer.mean()).select('Es_mean')

    evi = dataset_evi.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_evi = evi.select(['EVI_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                                            geometry=country_geom, scale=100000)
    dataset_evi = evi.select(['EVI_mean']).unitScale(ee.Number(minmax_evi.
    get(
        'EVI_mean_min')), ee.Number(minmax_evi.get('EVI_mean_max')))

    et = dataset_et.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_et = et.select(['Ec_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                          geometry=country_geom, scale=100000)
    dataset_et = et.select(['Ec_mean']).unitScale(ee.Number(minmax_et.
    get(
        'Ec_mean_min')), ee.Number(minmax_et.get('Ec_mean_max')))

    gpp = dataset_gpp.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_gpp = gpp.select(['GPP_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                    geometry=country_geom, scale=100000)
    dataset_gpp = gpp.select(['GPP_mean']).unitScale(ee.Number(minmax_gpp.
    get(
        'GPP_mean_min')), ee.Number(minmax_gpp.get('GPP_mean_max')))

    es = dataset_es.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)

    minmax_es = es.select(['Es_mean']).reduceRegion(reducer=ee.Reducer.minMax(),
                                                    geometry=country_geom, scale=100000)
    dataset_es = es.select(['Es_mean']).unitScale(ee.Number(minmax_es.
    get(
        'Es_mean_min')), ee.Number(minmax_es.get('Es_mean_max')))

    def set_time(img):
        return img.set('system:time_start', ee.Date(start.isoformat()).millis())

    stacked = ee.Image(dataset_evi).addBands(dataset_et).addBands(dataset_gpp).addBands(dataset_es)
    dataset_unitscale = (ee.ImageCollection([stacked])
                         .map(clip_collection)).map(set_time)

    dr_eco = xr.open_dataset(dataset_unitscale, engine='ee')

    return dr_eco


def slice_daily(start: datetime, end: datetime):
    merged_xr = xr.merge([get_climate(start=start, end=end), get_wildfire(start=start, end=end),
                          get_landcover(start=start), get_topography(start=start), get_ecology(start=start, end=end)])

    return merged_xr


if __name__=='__main__':
    start_time = datetime.fromisoformat('2001-01-01')
    end_time = datetime.fromisoformat('2016-12-15')

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
            limit += timedelta(days=15)

        while cur < limit:
            # Yield the slice (start, end_of_slice)
            yield cur, cur + timedelta(days=15)
            cur += timedelta(days=15)


    # get list of daily slices
    for day in daily_slices_gen(start_time, end_time, inclusive_end=False):
        print(day[0].isoformat(), day[1].isoformat())
        dataset_list.append(slice_daily(start=day[0], end=day[1]))

    dataset_combined = xr.combine_by_coords(dataset_list)
    dataset_filled = dataset_combined.fillna(0)
    print(dataset_combined)
    xr.Dataset.to_zarr(dataset_filled, store="/mnt/d/global_fire.zarr", mode="w-")
