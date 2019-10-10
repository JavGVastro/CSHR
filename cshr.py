from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#matplotlib inline
sns.set_color_codes()

nom_reg='Lag'


damiani_tab1_file = "J_A+A_604_A135_table2.dat.fits"
tab = Table.read(damiani_tab1_file)
tab
df = tab.to_pandas()
df.describe()


sns.pairplot(df,
             vars=["RVHalpha", "RV[NII]6584", "RV[SII]6717", "RV[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )

#mask=df['sigHalpha'] < df['sigHalpha'].mean()-4*df['sigHalpha'].std()


sns.pairplot(df,
             vars=["RVHalpha", "sigHalpha", "NormHalpha"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="blue"),
             diag_kws=dict(bins=20, color="blue"),
            )


sns.pairplot(df,
             vars=["RV[NII]6584", "sig[NII]6584", "Norm[NII]6584"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="red"),
             diag_kws=dict(bins=20, color="red"),
            )

sns.pairplot(df,
             vars=["RV[SII]6731", "sig[SII]6731", "Norm[SII]6731"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="green"),
             diag_kws=dict(bins=20, color="green"),
            )
