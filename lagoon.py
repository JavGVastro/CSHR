#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#matplotlib inline
sns.set_color_codes()


# In[2]:


nom_reg='Lag'


# In[3]:


damiani_tab1_file = "J_A+A_604_A135_table2.dat.fits"
tab = Table.read(damiani_tab1_file)
tab
df = tab.to_pandas()
df.describe()


# In[4]:


sns.pairplot(df,
             vars=["NormHalpha", "Norm[NII]6584", "Norm[SII]6717", "Norm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[5]:


sns.pairplot(df,
             vars=["RVHalpha", "RV[NII]6584", "RV[SII]6717", "RV[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[6]:


m=df['sigHalpha'] < df['sigHalpha'].mean()+4*df['sigHalpha'].std()


# In[7]:


sns.pairplot(df[m],
             vars=["sigHalpha", "sig[NII]6584", "sig[SII]6717", "RV[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# In[8]:


m=df['sigHalpha'] < df['sigHalpha'].mean()+4*df['sigHalpha'].std()


# In[9]:


sns.pairplot(df[m],
             vars=["RVHalpha", "sigHalpha", "NormHalpha"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="blue"),
             diag_kws=dict(bins=20, color="blue"),
            )


# In[10]:


sns.pairplot(df,
             vars=["RV[NII]6584", "sig[NII]6584", "Norm[NII]6584"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="red"),
             diag_kws=dict(bins=20, color="red"),
            )


# In[11]:


sns.pairplot(df,
             vars=["RV[SII]6731", "sig[SII]6731", "Norm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="green"),
             diag_kws=dict(bins=20, color="green"),
            )


# Halpha SF

# In[12]:


df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.RVHalpha, '_key': 1}
)


df1.describe()


# In[13]:


df2 = df1.copy()


# In[14]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[15]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[16]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]


# In[17]:


pairs.head()


# In[18]:



pairs.describe()


# In[19]:



pairs.corr()


# In[20]:



mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="blue")

ax.fig.set_size_inches(12, 12)


# In[21]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="blue")
ax.fig.set_size_inches(12, 12)


# In[22]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)


# In[23]:


pairs.s_class[pairs.s_class == 0] = 1


# In[24]:



for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[25]:


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
                 label=label, bins=20,color="blue", hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[26]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[27]:


ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4,color="blue")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.08*sgrid**(0.83), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# Modifications on RA and De reduction

# De

# In[28]:




df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.RVHalpha, '_key': 1}
)


df1.describe()


# In[29]:


df2 = df1.copy()


# In[30]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[31]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[32]:


pairs = pairs[(pairs.dDE > 0.0) ]


# In[33]:


pairs.head()


# In[34]:


pairs.describe()


# In[35]:


pairs.corr()


# In[36]:



mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.1, s=1, edgecolor='none',color="purple")
ax.fig.set_size_inches(12, 12)


# In[37]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.1, s=1, edgecolor='none',color="purple")
ax.fig.set_size_inches(12, 12)


# In[38]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)


# In[39]:


pairs.s_class[pairs.s_class == 0] = 1


# In[40]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[41]:


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
                 label=label, bins=20, color="purple",hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[42]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[43]:


ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4, color="purple")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.095*sgrid**(0.83), color="k", lw=0.5)
sgrid1 = np.logspace(2.5, 4.0)
ax.plot(sgrid1, 0.015*sgrid1**(1), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# RA

# In[44]:


df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.RVHalpha, '_key': 1}
)


df1.describe()


# In[45]:


df2 = df1.copy()


# In[46]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[47]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[48]:


pairs = pairs[(pairs.dRA > 0.0)]


# In[49]:


pairs.head()


# In[50]:


pairs.describe()


# In[51]:


pairs.corr()


# In[52]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.1, s=1, edgecolor='none',color="magenta")

ax.fig.set_size_inches(12, 12)


# In[53]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.1, s=1, edgecolor='none',color="magenta")
ax.fig.set_size_inches(12, 12)


# In[54]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)


# In[55]:


pairs.s_class[pairs.s_class == 0] = 1


# In[56]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[57]:


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
                 label=label, bins=20,color="magenta", hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[58]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[59]:


ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4, color="magenta")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.095*sgrid**(0.83), color="k", lw=0.5)
sgrid1 = np.logspace(2.5, 4.0)
ax.plot(sgrid1, 0.015*sgrid1**(1), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# [NII]6584 SF

# In[60]:


df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df['RV[NII]6584'], '_key': 1}
)


df1.describe()


# In[61]:


df2 = df1.copy()


# In[62]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[63]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[64]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]


# In[65]:


pairs.head()


# In[66]:


pairs.describe()


# In[67]:



pairs.corr()


# In[68]:



mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="red")
ax.fig.set_size_inches(12, 12)


# In[69]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="red")
ax.fig.set_size_inches(12, 12)


# In[70]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)


# In[71]:


pairs.s_class[pairs.s_class == 0] = 1


# In[72]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[73]:


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
                 label=label, bins=20,color="red", hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[74]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[75]:


ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4, color="red")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.4*sgrid**(2/3), color="k", lw=0.5)
sgrid1 = np.logspace(2.5, 4.0)
ax.plot(sgrid1, 0.015*sgrid1**(1), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# [SII]6731 SF

# In[76]:


df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df['RV[SII]6731'], '_key': 1}
)


df1.describe()


# In[77]:


df2 = df1.copy()


# In[78]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[79]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[80]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]


# In[81]:


pairs.head()


# In[82]:



pairs.describe()


# In[83]:


pairs.corr()


# In[84]:



mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="green")
ax.fig.set_size_inches(12, 12)


# In[85]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="green")
ax.fig.set_size_inches(12, 12)


# In[86]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)


# In[87]:


pairs.s_class[pairs.s_class == 0] = 1


# In[88]:



for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[89]:


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
                 label=label, bins=20, color="green",hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[90]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[91]:


ngroup = 500
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
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4, color="green")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.4*sgrid**(2/3), color="k", lw=0.5)
sgrid1 = np.logspace(2.5, 4.0)
ax.plot(sgrid1, 0.015*sgrid1**(1), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# Objects on interest

# In[92]:


types = ['HD 164536', '7 Sgr', 'Herschel 36', '9 Sgr', 'HD 164816', 'HD 164865', 'M8E-IR', 'HD 165052','HD 165246']
x_coords = [270.6609, 270.7129, 270.9180, 270.9685, 270.9869, 271.0634, 271.2244, 271.2940,271.5195]
y_coords = [-24.2554, -24.2825, -24.3785, -24.3607, -24.3126, -24.1834, -24.4448, -24.3986,-24.1955]


# In[93]:


points_of_interest = {
    "HD 164536": [270.6609, -24.2554],
    "7 Sgr": [270.7129, -24.2825],
    "Herschel 36": [270.9180, -24.3785],
    "9 Sgr": [270.9685, -24.3607],
    "HD 164816": [270.9869, -24.3126],
    "HD 164865": [271.0634, -24.1834],
    "M8E-IR": [271.2244, -24.4448],
    "HD 165052": [271.2940, -24.3986],
    "HD 165246": [271.5195, -24.1955],
}
def mark_points(ax):
    for label, c in points_of_interest.items():
        ax.plot(c[0], c[1], marker='+', markersize='12', color='k')


# Velocity maps

# Note: Logarithmic plot does't show the brightness differences

# In[94]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, 
                      s=0.0015*((df.NormHalpha)), 
                      c=df.RVHalpha 
                     )
    fig.colorbar(scat, ax=[ax])
    #mark_points(ax)
    #ax.set_facecolor('k')
    #ax.axis('equal')
    ax.set_aspect('equal', 'datalim')
    ax.invert_xaxis()


    
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='+', color='yellow')
    plt.text(x, y, type, fontsize=12)


# Emission Map

# In[95]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=np.log10(df.NormHalpha), cmap='inferno', vmin=3.5, vmax=5.5)
    fig.colorbar(scat, ax=ax).set_label("log10(F)")
    mark_points(ax)
    ax.set_title('H alpha brightness')
    ax.axis('equal')
    ax.axis([270.5, 271.7, -24.6, -24])
    ax.invert_xaxis()


# Density map

# In[96]:


def eden(R):
    """Approximate sii electron density from R=6717/6731"""
    RR = 1.0/R
    return 2489*(RR - 0.6971) / (2.3380 - RR)


# In[97]:


dfSii1 = pd.DataFrame(
    {'log_F': np.log10(df['Norm[SII]6717']), 
     'RV': df['RV[SII]6717'], 
     'sigma': df['sig[SII]6717'],
    }
).dropna()


# In[98]:


dfSii2 = pd.DataFrame(
    {'log_F': np.log10(df['Norm[SII]6731']), 
     'RV': df['RV[SII]6731'], 
     'sigma': df['sig[SII]6731'],
    }
).dropna()


# In[99]:


fSii1=df['Norm[SII]6717']
vSii1=df['RV[SII]6717']
sSii1=df['sig[SII]6717']


# In[100]:


fSii2=df['Norm[SII]6731']
vSii2=df['RV[SII]6731']
sSii2=df['sig[SII]6731']


# In[101]:


dfSii = pd.DataFrame(
    {'log_F': np.log10(fSii1 + fSii2), 
     'R12': fSii1/fSii2,
     'dV12': vSii1 - vSii2, 
     'V': (fSii1*vSii1 + fSii2*vSii2)/(fSii1 + fSii2),
     'sigma': np.sqrt((fSii1*sSii1**2 + fSii2*sSii2**2)/(fSii1 + fSii2)),
     'sigma12': sSii1/sSii2,
     'RAdeg': df.RAdeg,
     'DEdeg': df.DEdeg,
    }
).dropna()


# In[102]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=eden(dfSii.R12), cmap='Greys', vmin=0.0, vmax=1000.0)
    fig.colorbar(scat, ax=ax).set_label('$n_e$')
    #mark_points(ax)
    ax.invert_xaxis()
    #ax.set_aspect(2)
    ax.set_title('[S II] density')
    ax.axis('equal')


for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x, y, type, fontsize=15)


# Effective Layer thickness

# In[103]:


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, 
                      c=df['NormHalpha']/eden(dfSii.R12)**2, 
                      vmin=0.0, vmax=50.0, cmap='plasma_r')
    fig.colorbar(scat, ax=ax).set_label('$H$')
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('Effective layer thickness')
    ax.axis('equal')


# In[104]:


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, 
                      c=df['Norm[NII]6584']/df['NormHalpha'], 
                      cmap='hot_r')
    fig.colorbar(scat, ax=ax).set_label('$6583 / 6563$')
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('[N II] / H alpha ratio')
    ax.axis('equal')


# Velocity profiles

# In[105]:


df.describe()


# In[106]:


plt.scatter(df.RAdeg,df.DEdeg)
ax = plt.gca()
ax.invert_xaxis()
#vertical slit
#ax.axvline(271, color='k')
#ax.axvline(270.99, color='r')
#horizontl slit 0
ax.axhline(-24.1, color='k')
ax.axhline(-24.125, color='r')
#horizontl slit 1
ax.axhline(-24.2, color='k')
ax.axhline(-24.225, color='r')
#horizontl slit 2
ax.axhline(-24.3, color='k')
ax.axhline(-24.325, color='r')
#horizontl slit 3
ax.axhline(-24.4, color='k')
ax.axhline(-24.42, color='r')
#horizontl slit 4
ax.axhline(-24.5, color='k')
ax.axhline(-24.52, color='r')


# In[107]:


argo0 = df[(df.DEdeg < -24.1) & (df.DEdeg > -24.125)]
plt.figure(figsize=(10,2))
plt.scatter(argo0.RAdeg,argo0.RVHalpha,s=0.0015*(argo0.NormHalpha))
ax = plt.gca()
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])
#ax.invert_xaxis()


# In[108]:


argo1 = df[(df.DEdeg < -24.2) & (df.DEdeg > -24.225)]
plt.figure(figsize=(10,2))
plt.scatter(argo1.RAdeg,argo1.RVHalpha,s=0.0015*(argo1.NormHalpha))
ax = plt.gca()
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])
#ax.invert_xaxis()


# In[109]:


argo2 = df[(df.DEdeg < -24.3) & (df.DEdeg > -24.325)]
plt.figure(figsize=(10,2))
plt.scatter(argo2.RAdeg,argo2.RVHalpha,s=0.0015*(argo2.NormHalpha))
ax = plt.gca()
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])
#ax.invert_xaxis()


# In[110]:


argo3 = df[(df.DEdeg < -24.4) & (df.DEdeg > -24.42)]
plt.figure(figsize=(10,2))
plt.scatter(argo3.RAdeg,argo3.RVHalpha,s=0.0015*(argo3.NormHalpha))
ax = plt.gca()
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])
#ax.invert_xaxis()


# In[111]:


argo4 = df[(df.DEdeg < -24.5) & (df.DEdeg > -25.42)]
plt.figure(figsize=(10,2))
plt.scatter(argo4.RAdeg,argo4.RVHalpha,s=0.0015*(argo4.NormHalpha))
ax = plt.gca()
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])
#ax.invert_xaxis()


# In[112]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df.RVHalpha, c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[113]:


fig, ax = plt.subplots(figsize=(5, 3))
scat = plt.scatter(df.DEdeg, df.RVHalpha, c=-df.RAdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="gist_ncar")
fig.colorbar(scat, ax=ax).set_label("RA")


# Fit a cubic function to V(Ha) vs RA to try and remove the large-scale trend. 

# In[114]:


pRA = np.poly1d(np.polyfit(df.RAdeg, df.RVHalpha, 3))


# In[172]:


df.RVHalpha


# In[115]:


print(pRA)


# Check that the function is a good fit

# In[116]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df.RVHalpha, c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
RAgrid = np.linspace(df.RAdeg.min(), df.RAdeg.max())
ax.plot(RAgrid, pRA(RAgrid), c="k", ls="--")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# Subtract the trend from the data.

# In[117]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df.RVHalpha - pRA(df.RAdeg), c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[118]:


df1 = pd.DataFrame({'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.RVHalpha - pRA(df.RAdeg), '_key': 1})
df1.describe()


# In[119]:


df2 = df1.copy()


# In[120]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[121]:



pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[122]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs.head()


# In[123]:


pairs.describe()


# In[124]:


pairs.corr()


# In[125]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="blue")
ax.fig.set_size_inches(12, 12)


# In[126]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="blue")
ax.fig.set_size_inches(12, 12)


# In[127]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1


# In[128]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[129]:


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


# In[130]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[131]:


ngroup = 500
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
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.095*sgrid**(0.8), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# In[132]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, 
                      s=0.0015*((df.NormHalpha)), 
                      c=df.RVHalpha - pRA(df.RAdeg)
                     )
    fig.colorbar(scat, ax=[ax])
    ax.set_aspect('equal', 'datalim')
    ax.invert_xaxis()


    
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='+', color='yellow')
    plt.text(x, y, type, fontsize=12)


# [NII]6584 polynomial fit

# In[133]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df['RV[NII]6584'], c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[134]:


pRA = np.poly1d(np.polyfit(df.RAdeg, df['RV[NII]6584'], 3))
print(pRA)


# In[135]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg,df['RV[NII]6584'], c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
RAgrid = np.linspace(df.RAdeg.min(), df.RAdeg.max())
ax.plot(RAgrid, pRA(RAgrid), c="k", ls="--")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[136]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df['RV[NII]6584'] - pRA(df.RAdeg), c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[137]:


df1 = pd.DataFrame({'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df['RV[NII]6584'] - pRA(df.RAdeg), '_key': 1})
df1.describe()


# In[138]:


df2 = df1.copy()


# In[139]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[140]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[141]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs.head()


# In[142]:


pairs.describe()


# In[143]:


pairs.corr()


# In[144]:




mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="darkred")
ax.fig.set_size_inches(12, 12)


# In[145]:




mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="darkred")
ax.fig.set_size_inches(12, 12)


# In[146]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1


# In[147]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[148]:


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
                 label=label, bins=20, color="darkred",hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[149]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[150]:


ngroup = 500
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
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.3, color="darkred")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.225*sgrid**(0.77), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# In[151]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, 
                      s=0.0015*((df.NormHalpha)), 
                      c=df['RV[NII]6584'] - pRA(df.RAdeg)
                     )
    fig.colorbar(scat, ax=[ax])
    ax.set_aspect('equal', 'datalim')
    ax.invert_xaxis()


    
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='+', color='yellow')
    plt.text(x, y, type, fontsize=12)


# In[152]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df['RV[SII]6731'], c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[153]:


pRA = np.poly1d(np.polyfit(df.RAdeg, df['RV[SII]6731'], 3))
print(pRA)


# In[154]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df['RV[SII]6731'], c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
RAgrid = np.linspace(df.RAdeg.min(), df.RAdeg.max())
ax.plot(RAgrid, pRA(RAgrid), c="k", ls="--")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[155]:


fig, ax = plt.subplots(figsize=(10, 3))
scat = plt.scatter(df.RAdeg, df['RV[SII]6731'] - pRA(df.RAdeg), c=df.DEdeg, s=50*(np.log10(df.NormHalpha) - 3), alpha=0.3, cmap="coolwarm")
fig.colorbar(scat, ax=ax).set_label("Dec")
ax.set(xlim=[df.RAdeg.max(),df.RAdeg.min()])


# In[156]:


df1 = pd.DataFrame({'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df['RV[SII]6731'] - pRA(df.RAdeg), '_key': 1})
df1.describe()


# In[157]:


df2 = df1.copy()


# In[158]:


pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.head()


# In[159]:


pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)


# In[160]:


pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs.head()


# In[161]:


pairs.describe()


# In[162]:


pairs.corr()


# In[163]:




mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=pairs[mask], alpha=0.2, s=1, edgecolor='none',color="darkgreen")
ax.fig.set_size_inches(12, 12)


# In[164]:


mask = (pairs.log_s > 0.0) & (pairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=pairs[mask], alpha=0.2, s=1, edgecolor='none', color="darkgreen")
ax.fig.set_size_inches(12, 12)


# In[165]:


pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1


# In[166]:


for j in range(7):
    print()
    print("s_class =", j)
    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())


# In[167]:


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
                 label=label, bins=20, color="darkgreen",hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()


# In[168]:


print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')


# In[169]:


ngroup = 500
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
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.3, color="darkgreen")
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.225*sgrid**(0.76), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None


# In[170]:


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, 
                      s=0.0015*((df.NormHalpha)), 
                      c=df['RV[SII]6731'] - pRA(df.RAdeg)
                     )
    fig.colorbar(scat, ax=[ax])
    ax.set_aspect('equal', 'datalim')
    ax.invert_xaxis()


    
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='+', color='yellow')
    plt.text(x, y, type, fontsize=12)

