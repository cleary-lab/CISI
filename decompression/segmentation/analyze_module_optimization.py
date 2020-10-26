import numpy as np
import glob,os
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

a = 10**np.linspace(-1,2,10)
b = 10**np.linspace(-2,1,10)
x = np.zeros((10,10))
FP = glob.glob(os.path.join('Logs/Optimize.o19226927.*'))
for fp in FP:
	f = open(fp)
	L = f.readlines()
	f.close()
	if len(L) > 0:
		i = [float(l.strip().split('\t')[-1]) for l in L if 'alpha_cond' in l][0]
		j = [float(l.strip().split('\t')[-1]) for l in L if 'alpha_corr' in l][0]
		x[np.where(abs(a-i) < 1e-4)[0][0],np.where(abs(b-j) < 1e-4)[0][0]] = float(L[-2].strip().split()[-1])


astr = ['%.2f'%i for i in a]
df = pd.DataFrame(x,columns=b,index=astr)
sns_plt = sns.heatmap(df,mask=(df == 0),annot=True)
plt.savefig('alpha_grid.png')
plt.close()