### EDA

## import libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

import warnings, gc
warnings.filterwarnings('ignore')


# * Reduce data size for memory and disk (50GB -> 5GB)
#   * Processed dataset to transform float data types to int data types (parquet format)
#   * 1. Reduce data types
#     * Column customer_ID - Reduce 64 bytes to 4 bytes
#     * Column S_2 - Reduce 10 bytes to 3 bytes
#       - df_train['S_2'] = pd.to_datetime(df_train['S_2'])
#       - df_test['S_2'] = pd.to_datetime(df_test['S_2'])
#     * 11 Categorical Columns - Reduce 88 bytes to 11 bytes!
#       - ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
#     * 177 Numeric Columns - Reduce 1416 bytes to 353 bytes
#   * 2. Choose file format
#     * parquet (almost 5GB)
#   * 3. Choose Multiple Files or Not(one large file)
#     * trouble processing the one large file -> processing in chunks and saving the data to disk as separate files
# * 'conda install -c conda-forge pyarrow fastparquet' in terminal for parquet


## import and read the dataset
df_train = pd.read_parquet('/Users/mac/Desktop/Amex/data/train.parquet')
df_train = df_train.groupby('customer_ID').tail(1).set_index('customer_ID')

df_test = pd.read_parquet('/Users/mac/Desktop/Amex/data/test.parquet')
df_test = df_test.groupby('customer_ID').tail(1).set_index('customer_ID')

train_labels = pd.read_csv('/Users/mac/Desktop/Amex/data/train_labels.csv')

df_sub = pd.read_csv("/Users/mac/Desktop/Amex/data/sample_submission.csv")

print(df_train.shape)
print(df_test.shape)

print(df_train.head())
print(df_train.info())

# 
feat_Delinquency = [c for c in df_train.columns if c.startswith('D_')]
feat_Spend = [c for c in df_train.columns if c.startswith('S_')]
feat_Payment = [c for c in df_train.columns if c.startswith('P_')]
feat_Balance = [c for c in df_train.columns if c.startswith('B_')]
feat_Risk = [c for c in df_train.columns if c.startswith('R_')]
print(f'Total number of Delinquency variables: {len(feat_Delinquency)}')
print(f'Total number of Spend variables: {len(feat_Spend)}')
print(f'Total number of Payment variables: {len(feat_Payment)}')
print(f'Total number of Balance variables: {len(feat_Balance)}')
print(f'Total number of Risk variables: {len(feat_Risk)}')


## EDA
labels=['Delinquency', 'Spend','Payment','Balance','Risk']
values= [len(feat_Delinquency), len(feat_Spend),len(feat_Payment), len(feat_Balance),len(feat_Risk)]

# feature distribution
fig_1 = go.Figure()
fig_1.add_trace(go.Pie(values = values,labels = labels,hole = 0.6, 
                     hoverinfo ='label+percent'))
fig_1.update_traces(textfont_size = 12, hoverinfo ='label+percent',textinfo ='label', 
                  showlegend = False,marker = dict(colors =["#70d6ff","#ff9770"]),
                  title = dict(text = 'Feature Distribution'))  
fig_1.show()

# 
df_train.isna().sum()

background_color = 'white'
missing = pd.DataFrame(columns = ['% Missing values'],data = df_train.isnull().sum()/len(df_train))
fig = plt.figure(figsize = (20, 60),facecolor=background_color)
gs = fig.add_gridspec(1, 2)
gs.update(wspace = 0.5, hspace = 0.5)
ax0 = fig.add_subplot(gs[0, 0])
for s in ["right", "top","bottom","left"]:
    ax0.spines[s].set_visible(False)
sns.heatmap(missing,cbar = False,annot = True,fmt =".2%", linewidths = 2, vmax = 1, ax = ax0)
plt.show()