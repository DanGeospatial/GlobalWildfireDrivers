import xarray as xr
import dask
import matplotlib.pyplot as plt

dask.config.set(scheduler='synchronous')

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    ds = xr.open_zarr("/mnt/d/global_fire.zarr")
    print(ds)

ds.heat_index_mean.isel(time=0).plot(x="lon", y="lat")
plt.show()