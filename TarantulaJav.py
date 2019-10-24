#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import seaborn as sns


# Note: import downsample

# In[2]:


def downsample(images, mask, weights=None, verbose=False, mingood=1):
    """
    Resample (average) a list of 2d images at 2x2, taking account of a logical mask

    Now optionally use a weights array, and resample that too (adding, not averaging)
    """
    # Construct slices for even and odd elements, respectively
    # e, o = np.s_[:,-1,2], np.s_[1,:,2] # Just learnt about the np.s_() function!
    e, o = slice(None,-1,2), slice(1,None,2)

    # Find the number of good sub-pixels in each new pixel
    ngood = mask[e,e].astype(int) + mask[o,e].astype(int)             + mask[e,o].astype(int) + mask[o,o].astype(int)
    

    newmask = ngood >= mingood
    # # Resample the mask
    # # newmask is True if any of the 4 sub-pixels are true
    # newmask = mask[e,e] | mask[o,e] | mask[e,o] | mask[o,o]

    if weights is None:
        # now resample the images
        newimages = [
            np.where(
                newmask,      # Check that we have at least 1 good pixel
                # Take the mean of all the good sub-pixels
                (image[e,e]*mask[e,e] + image[o,e]*mask[o,e]
                 + image[e,o]*mask[e,o] + image[o,o]*mask[o,o]) / ngood,
                0.0                 # Avoid NaNs if we have no good pixels
                )
            for image in images]
    else:
        newweights = (weights[e,e]*mask[e,e] + weights[o,e]*mask[o,e]
			+ weights[e,o]*mask[e,o] + weights[o,o]*mask[o,o])
        newimages = [
            np.where(
                newweights > 0.0,      # Check that we have at least 1 good pixel
                # Take the mean of all the good sub-pixels
                (image[e,e]*mask[e,e]*weights[e,e] +
                 image[o,e]*mask[o,e]*weights[o,e] +
                 image[e,o]*mask[e,o]*weights[e,o] +
                 image[o,o]*mask[o,o]*weights[o,o]
                 ) / newweights,
                0.0                 # Avoid NaNs if we have no good pixels
                )
            for image in images]

    if verbose:
        print("Fraction of good pixels: old = {:.2f}, new = {:.2f}".format(
            float(mask.sum())/mask.size, float(newmask.sum())/newmask.size))
    # Of course, we will get garbage where ngood=0, but that doesn't
    # matter since newmask will be False for those pixels
    if weights is None:
        return newimages, newmask
    else:
        # New option to bin weights too
        return newimages, newmask, newweights


# In[3]:


hdulist = fits.open("GAUS_Ha6562.8_060_Will.fits")


# Coordinates: Will

# In[4]:


vhdu = hdulist[2]


# In[5]:


w = WCS(vhdu)


# In[6]:


w = w.celestial


# In[7]:


w.pixel_to_world


# In[8]:


ny, nx = vhdu.shape


# In[9]:


X, Y = np.meshgrid( np.arange(nx), np.arange(ny))


# In[10]:


c = w.pixel_to_world(X, Y)


# Data analysis: Memory RAM needed for calculation

# Note: check equation

# In[11]:


ds=0
n=2**ds
x=len(X)//n
y=len(Y)//n
xy2=(x*y)**2
RAM=xy2*8/10**9
print(x,y,x*y,(x*y)**2)
print( str(RAM) +  ' GB needed to manage the SF')


# Downsample

# In[12]:


mingoods = [2,2,2,2]


# RA

# In[13]:


radeg=c.ra.degree
m=pd.notna(radeg)


# In[14]:


for mingood in zip(mingoods):
 [radeg], m= downsample([radeg], m, weights=None, mingood=mingood)


# Dec

# In[15]:


decdeg=c.dec.degree
m=pd.notna(decdeg)


# In[16]:


for mingood in zip(mingoods):
 [decdeg], m= downsample([decdeg], m, weights=None, mingood=mingood)


# Velocity Downsample

# In[17]:


RV=hdulist[2].data
m=pd.notna(RV)


# In[18]:


for mingood in zip(mingoods):
 [RV], m= downsample([RV], m, weights=None, mingood=mingood)


# Mask

# In[19]:


RVm=RV>0
radegm=radeg>0
decdegm=abs(decdeg)>0


# In[20]:


rvf=RV[RVm]
raf=radeg[radegm]
decf=decdeg[decdegm]


# Main DataFrame

# In[21]:


RApd=pd.DataFrame(raf)
DEpd=pd.DataFrame(decf)
RVpd=pd.DataFrame(rvf)


# In[22]:


df=pd.concat([RApd,DEpd,RVpd],axis=1)
df.columns = ['RAdeg', 'DEdeg', 'RVHalpha']
df.describe()


# Adjust color scale for a better fit

# In[50]:


dataRV=hdulist[2].data
plt.figure(1)
plt.imshow(dataRV, cmap='gray')
plt.gca().invert_yaxis()
plt.figure(2)
plt.imshow(np.log10(RV), cmap='gray')
plt.gca().invert_yaxis()


# Velocity Profiles

# In[24]:


fig, ax = plt.subplots(figsize=(5, 4))
scat = plt.scatter(df.RAdeg, df.RVHalpha, c=df.DEdeg, alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[25]:


fig, ax = plt.subplots(figsize=(5, 4))
scat = plt.scatter(df.DEdeg, df.RVHalpha, c=-df.RAdeg, alpha=0.3, cmap="gist_ncar")
fig.colorbar(scat, ax=ax).set_label("RA")
ax.set(xlim=[df.DEdeg.max()-0.003,df.DEdeg.min()-0.001])


# Structure Function

# In[26]:


df1 = pd.DataFrame({'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.RVHalpha, '_key': 1})
df1.describe()


# In[27]:


df2 = df1.copy()


# In[28]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[29]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[30]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs.head()


# In[31]:


pairs.describe()


# In[32]:


pairs.corr()


# In[33]:


mask = (pairs.log_s > 0) & (pairs.log_dV2 > -4)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="blue")
ax.fig.set_size_inches(12, 12)


# In[34]:


mask = (pairs.log_s > 0) & (pairs.log_dV2 > -4)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="blue", xlim=[0.05, 2.26])
ax.fig.set_size_inches(12, 12)


# In[35]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1


# In[36]:


for j in range(5):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[37]:


sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 5), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=True, kde=False, ax=ax,
                 label=label, bins=20, color="blue",hist_kws=dict(range=[-4.0, 4.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-4.0, 4.0])
    ax.legend(loc='upper left')
sns.despine()


# In[38]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[39]:


ngroup = 2000
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.3, color="blue")
sgrid = np.logspace(0.3, 1.65)
ax.plot(sgrid, 38.5*sgrid**(0.8), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# In[ ]:




