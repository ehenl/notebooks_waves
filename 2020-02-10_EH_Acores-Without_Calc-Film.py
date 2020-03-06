#!/usr/bin/env python
# coding: utf-8

# In[1]:


## path for modules

import sys
#sys.path.insert(0,"/home/albert/lib/python")

import numpy as np
import xarray as xr
#import glob

from netCDF4 import Dataset

#sys.path.insert(0,"/home/henelle/Notebooks/git/xscale")
#import xscale
#import xscale.spectral.fft as xfft

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy.ma as ma

import matplotlib.cm as mplcm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

seq_cmap = mplcm.Blues
div_cmap = mplcm.seismic

import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import date

#import seaborn as sns
#sns.set(color_codes=True)

#from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
#from math import cos, pi 

import pandas as pd



# In[2]:


## Dataset

#dirfilt="/home/henelle/Notebooks/Data/"
dirfilt="/mnt/meom/workdir/henelle/Notebooks/Data/"


# In[3]:


## JFM & JAS files

tfileJFM = dirfilt + 'ACO_JFM_filtered.nc'
tfileJAS = dirfilt + 'ACO_JAS_filtered.nc'


# params = {'font.weight':    'normal',
#           'font.size':       int(14),
#           'figure.titlesize': 'large',
#           'legend.fontsize': int(12),
#           'xtick.labelsize': int(14),
#           'ytick.labelsize': int(14),
#           'axes.labelsize':  int(14)}
# mpl.rcParams.update(params)

# In[4]:


fJFM = xr.open_dataset(tfileJFM)
fJAS = xr.open_dataset(tfileJAS)


# In[5]:


navlat = fJFM['lat']
navlon = fJFM['lon']


# In[6]:


## RV variables

# JFM
rvnotideJFM = fJFM['rv_notides_inst']
rv_fromSSH_ave24h_JFM = fJFM['rv_fromSSH_ave24h']

# JAS
rvnotideJAS = fJAS['rv_notides_inst']
rv_fromSSH_ave24h_JAS = fJAS['rv_fromSSH_ave24h']


# In[7]:


f = 1e-4


# In[8]:


## Normalizing with f

rv_notide_over_f_JFM = rvnotideJFM/f
rv_over_f_24h_JFM = rv_fromSSH_ave24h_JFM/f

rv_notide_over_f_JAS = rvnotideJAS/f
rv_over_f_24h_JAS = rv_fromSSH_ave24h_JAS/f


# In[9]:


print(np.min(rv_notide_over_f_JFM));print(np.max(rv_notide_over_f_JFM))
print(np.min(rv_over_f_24h_JFM));print(np.max(rv_over_f_24h_JFM))


# In[10]:


def plot_surf(data1,data2,lon,lat,i,vmin,vmax,cmap,title,date,season):
    
    fig = plt.figure(figsize=(10,9))
    
    ax1 = fig.add_subplot(121,projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(122,projection=ccrs.PlateCarree())
    
    # Adjust subplots
    plt.subplots_adjust(hspace=0.0,wspace=0.0) # 0.025
    
    norm_fld = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    
    pcolor1 = ax1.pcolormesh(lon,lat,data1,cmap=cmap,vmin=vmin,vmax=vmax,norm = norm_fld)
    pcolor2 = ax2.pcolormesh(lon,lat,data2,cmap=cmap,vmin=vmin,vmax=vmax,norm = norm_fld)
    
    # Add the Azores
    land = cfeature.GSHHSFeature(scale='intermediate',
                                 levels=[1],
                                 facecolor='dimgray')
    ax1.add_feature(land)
    ax2.add_feature(land)
    
    
    # Colorbar ax1
    cax1,kw1   = mpl.colorbar.make_axes(ax1,location='bottom',pad=0.05,shrink=0.8)
    out1       = fig.colorbar(pcolor1,cax=cax1,extend='both',**kw1)
    out1.set_label('$\zeta/f$',size=14)
    out1.ax.tick_params(labelsize=14)
    xticks_ax1 = np.linspace(vmin,vmax,num=5)
    out1.set_ticks(xticks_ax1)
    
    # Colorbar ax2
    cax2,kw2   = mpl.colorbar.make_axes(ax2,location='bottom',pad=0.05,shrink=0.8)
    out2       = fig.colorbar(pcolor2,cax=cax2,extend='both',**kw2)
    out2.set_label('$\zeta/f$',size=14)
    out2.ax.tick_params(labelsize=14)
    xticks_ax2 = np.linspace(vmin,vmax,num=5)
    out2.set_ticks(xticks_ax2)
    
    # Grid    
    gl1            = ax1.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,lw=1,color='gray',alpha=0.8, ls='--')
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xlabel_style = {'size': 10, 'color': 'dimgray'}
    gl1.ylabel_style = {'size': 10, 'color': 'dimgray'}
    gl1.xlabels_top = False
    gl1.ylabels_right = False
    gl2            = ax2.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,lw=1,color='gray',alpha=0.8, ls='--')
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.xlabel_style = {'size': 10, 'color': 'dimgray'}
    gl2.ylabel_style = {'size': 10, 'color': 'dimgray'}
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.ylabels_left = False
    
    # Adjusting axes
    ax1.set_xlim((-36, -26))
    ax1.set_ylim((25, 40))
    ax2.set_xlim((-36, -26))
    ax2.set_ylim((25, 40))

    ts = pd.to_datetime(str(date))
    d = ts.strftime('%d/%m/%y %H:%M')
    
    ax1.set_title('$\zeta/f$ no tide',size=14, y=1.05)
    ax2.set_title('$\zeta/f$ 24h average',size=14, y=1.05)
    plt.suptitle(title+'; '+str(d),size=22,y=1.0)
    
    plt.savefig('./animation_files/file_'+season+'_'+str(str(i).zfill(4))+'.png',dpi=100,
                bbox_inches='tight',pad_inches=0.1)
    plt.clf()
    


# ### JFM Winter

# In[11]:


n = len(rv_notide_over_f_JFM[:,0,0]) # number of time steps
for i in range(n):
    plot_surf(rv_notide_over_f_JFM[i,:,:],rv_over_f_24h_JFM[i,:,:],navlon,navlat,i,
              vmin=-0.3,vmax=0.3,cmap='RdYlBu_r',title='$\zeta/f$ evolution JFM',
              date=str(fJFM.time.values[i]),season='JFM')


# ### JAS Summer

# In[12]:


#n = len(rv_notide_over_f_JFM[:,0,0]) # number of time steps


# In[13]:


n = len(rv_notide_over_f_JFM[:,0,0]) # number of time steps
for i in range(n):
    plot_surf(rv_notide_over_f_JAS[i,:,:],rv_over_f_24h_JAS[i,:,:],navlon,navlat,i,
              vmin=-0.3,vmax=0.3,cmap='RdYlBu_r',title='$\zeta/f$ evolution JAS',
              date=str(fJAS.time.values[i]),season='JAS')


# In[14]:


print(np.shape(rv_notide_over_f_JFM))
print(np.shape(rv_over_f_24h_JFM))


# In[15]:


397/6


# In[16]:


print(np.min(rv_notide_over_f_JAS));print(np.max(rv_notide_over_f_JAS))
print(np.min(rv_over_f_24h_JAS));print(np.max(rv_over_f_24h_JAS))


# In[ ]:





# In[ ]:




