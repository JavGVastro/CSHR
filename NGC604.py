#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_table('datos604.txt')
df.describe()


# In[3]:


m=(df.RA<145)&(df.RA>60)
df=df[m]
df.describe()


# In[4]:


#df.NormHalpha=df.NormHalpha[(df.NormHalpha<df.NormHalpha.mean()+3*df.NormHalpha.std())&(df.NormHalpha>df.NormHalpha.mean()-3*df.NormHalpha.std())]
#df['Norm[NII]6584']=df['Norm[NII]6584'][(df['Norm[NII]6584']<df['Norm[NII]6584'].mean()+3*df['Norm[NII]6584'].std())&(df['Norm[NII]6584']>df['Norm[NII]6584'].mean()-3*df['Norm[NII]6584'].std())]
#df['Norm[SII]6731']=df['Norm[SII]6731'][(df['Norm[SII]6731']<df['Norm[SII]6731'].mean()+3*df['Norm[SII]6731'].std())&(df['Norm[SII]6731']>df['Norm[SII]6731'].mean()-3*df['Norm[NII]6584'].std())]


# In[5]:


sns.pairplot(df,
             vars=["NormHalpha", "Norm[NII]6584", "Norm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[6]:


#df.RVHalpha=df.RVHalpha[(df.RVHalpha<df.RVHalpha.mean()+3*df.RVHalpha.std())&(df.RVHalpha>df.RVHalpha.mean()-3*df.RVHalpha.std())]
#df['RV[NII]6584']=df['RV[NII]6584'][(df['RV[NII]6584']<df['RV[NII]6584'].mean()+3*df['RV[NII]6584'].std())&(df['RV[NII]6584']>df['RV[NII]6584'].mean()-3*df['RV[NII]6584'].std())]
#df['RV[SII]6731']=df['RV[SII]6731'][(df['RV[SII]6731']<df['RV[SII]6731'].mean()+3*df['RV[SII]6731'].std())&(df['RV[SII]6731']>df['RV[SII]6731'].mean()-3*df['RV[SII]6731'].std())]


# In[7]:


sns.pairplot(df,
             vars=["RVHalpha", "RV[NII]6584", "RV[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[8]:


#m=df['SigmHalpha'] < df['SigmHalpha'].mean()+3*df['SigmHalpha'].std()


# In[9]:


sns.pairplot(df,
             vars=["SigmHalpha", "Sigm[NII]6584", "Sigm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[10]:


sns.pairplot(df,
             vars=["NormHalpha","RVHalpha", "SigmHalpha"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="blue"),
             diag_kws=dict(bins=20, color="blue"),
            )


# In[28]:


sns.pairplot(df,
             vars=["Norm[NII]6584","RV[NII]6584", "Sigm[NII]6584"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="red"),
             diag_kws=dict(bins=20, color="red"),
            )


# In[29]:


sns.pairplot(df,
             vars=["Norm[SII]6731","RV[SII]6731", "Sigm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="green"),
             diag_kws=dict(bins=20, color="green"),
            )


# Velocity Map

# In[13]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(10, 4))
    scat = ax.scatter(df.RA, df.DE, 
                      s=(0.085*(df.NormHalpha)), 
                      c=df.RVHalpha,
                     )
    fig.colorbar(scat, ax=[ax])
    #ax.set(xlim=[65.0, 145.0])
    #mark_points(ax)
    #ax.set_facecolor('k')
    #ax.axis('equal')
    #ax.set_aspect('equal')
    #ax.invert_xaxis()


# Intensity Map

# In[14]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(10, 4))
    scat = ax.scatter(df.RA, df.DE, 
                      c=np.log10(df.NormHalpha), s=200,  cmap='inferno', 
                     )
    fig.colorbar(scat, ax=[ax])
    ax.set(xlim=[65.0, 145.0])
    #mark_points(ax)
    ax.set_facecolor('k')
    #ax.axis('equal')
    #ax.set_aspect('equal')
    #ax.invert_xaxis()


# Sigma Map

# In[15]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(10, 4))
    scat = ax.scatter(df.RA, df.DE, 
                      s=(0.085*(df.NormHalpha)), 
                      c=df.SigmHalpha, cmap='viridis',
                     )
    fig.colorbar(scat, ax=[ax])
    ax.set(xlim=[65.0, 145.0])
    #mark_points(ax)
    ax.set_facecolor('k')
    #ax.axis('equal')
    #ax.set_aspect('equal')
    #ax.invert_xaxis()


# Density  Map

# In[16]:


def eden(R):
    """Approximate sii electron density from R=6717/6731"""
    RR = 1.0/R
    return 2489*(RR - 0.6971) / (2.3380 - RR)


# In[17]:


fSii1=df['Norm[SII]6717']
vSii1=df['RV[SII]6717']
sSii1=df['Sigm[SII]6717']

fSii2=df['Norm[SII]6731']
vSii2=df['RV[SII]6731']
sSii2=df['Sigm[SII]6731']


# In[18]:


dfSii = pd.DataFrame(
    {'log_F': np.log10(fSii1 + fSii2), 
     'R12': fSii1/fSii2,
     'dV12': vSii1 - vSii2, 
     'V': (fSii1*vSii1 + fSii2*vSii2)/(fSii1 + fSii2),
     'sigma': np.sqrt((fSii1*sSii1**2 + fSii2*sSii2**2)/(fSii1 + fSii2)),
     'sigma12': sSii1/sSii2,
     'RAdeg': df.RA,
     'DEdeg': df.DE,
    }
)


# In[19]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RA, df.DE, s=150, c=eden(dfSii.R12), cmap='Greys', vmin=0.0, vmax=500.0)
    fig.colorbar(scat, ax=ax).set_label('$n_e$')
    #mark_points(ax)
    ax.invert_xaxis()
    ax.set(xlim=[65.0, 145.0])
    ax.set_title('[S II] density')
   


# In[20]:


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RA, df.DE, s=100, 
                      c=df['NormHalpha']/eden(dfSii.R12)**2, 
                      vmin=0.0, vmax=50.0, cmap='plasma_r')
    fig.colorbar(scat, ax=ax).set_label('$H$')
    #mark_points(ax)
    ax.invert_xaxis()
    #ax.set_aspect(2)
    ax.set_title('Effective layer thickness')
    ax.set(xlim=[65.0, 145.0])


# In[21]:


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RA, df.DE, s=100, 
                      c=df['Norm[NII]6584']/df['NormHalpha'], 
                      cmap='hot_r', vmin=0.0, vmax=0.6)
    fig.colorbar(scat, ax=ax).set_label('$6583 / 6563$')
    #mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('[N II] / H alpha ratio')
    ax.set(xlim=[65.0, 145.0])


# In[30]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RA, df.RVHalpha, c=df.DE, s=df.NormHalpha*.09, alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
#ax.set(xlim=[65.0, 145.0])


# In[23]:


fig, ax = plt.subplots(figsize=(5, 3))
scat = plt.scatter(df.DE, df.RVHalpha, c=df.RA, s=0.06*(df.NormHalpha), alpha=0.3, cmap="gist_ncar", vmin=65.0, vmax=145.0,)
fig.colorbar(scat, ax=ax).set_label("RA")


# In[66]:


idx = np.isfinite(df.RA) & np.isfinite(df.RVHalpha)


# In[67]:


pRA = np.poly1d(np.polyfit(df.RA[idx], df.RVHalpha[idx], 3))
print(pRA)


# In[68]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RA, df.RVHalpha, c=df.DE, s=.09*(df.NormHalpha), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
RAgrid = np.linspace(df.RA.min(), df.RA.max())
ax.plot(RAgrid, pRA(RAgrid), c="k", ls="--")


# In[69]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RA, df.RVHalpha - pRA(df.RA), c=df.DE, s=.09*(df.NormHalpha), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
#ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[70]:


df1 = pd.DataFrame({'RA': df.RA, 'DE': df.DE, 'V': df.RVHalpha - pRA(df.RA), '_key': 1})
df1.describe()


# In[71]:


df2 = df1.copy()


# In[72]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[80]:


pairs.loc[:, 'dDE'] = (pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = (pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[81]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs.head()


# In[82]:


pairs.describe()


# In[83]:


pairs.corr()


# In[84]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="blue")
ax.fig.set_size_inches(12, 12)


# In[85]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="blue")
ax.fig.set_size_inches(12, 12)


# In[86]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1


# In[87]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[88]:


sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
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
                 label=label, bins=20, color="blue",hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[89]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[92]:


ngroup = 700
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
#sgrid = np.logspace(1.0, 3.0)
#ax.plot(sgrid, 0.095*sgrid**(0.8), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

