#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('marketing_campaign.csv', sep='\t')
import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected = True)


# In[8]:


df.head()


# In[9]:


cols = ["Year_Birth", "Education", "Income", "MntWines"]

filtered_df = df[cols]
filtered_df


# In[15]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas-profiling')


# In[16]:


import pandas_profiling as pp
pp.ProfileReport(df)


# In[13]:


import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import numpy as np


# In[15]:


dist = df['Year_Birth'].value_counts()
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Year of Birth')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors= colors, line=dict(color='#000000', width=2)))
fig.show()


# In[10]:


dist = df['Education'].value_counts()
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Education')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# In[12]:


dist = df['Income'].value_counts()
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Income')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# In[13]:


dist = df['MntWines'].value_counts()
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='MntWines')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# In[14]:


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }
import plotly.graph_objects as go
dfNew = df.corr()
fig = go.Figure(data=go.Heatmap(df_to_plotly(dfNew)))
fig.show()


# In[15]:


fig = px.scatter(df, x='Year_Birth', y='MntWines')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Age and Wine')
fig.show()


# ### fig = px.scatter(df, x='Education', y='MntWines')
# fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
#                   marker_line_width=1.5)
# fig.update_layout(title_text='Education and Wine')
# fig.show()

# In[16]:


fig = px.scatter(df, x='Income', y='MntWines')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Income and Wine')
fig.show()


# In[21]:


fig = px.box(df, x='Income', y='MntWines')
fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Income and Wine')
fig.show()


# In[24]:


plot = sns.boxplot(x='Income',y="MntWines",data=df)


# In[17]:


import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('marketing_campaign.csv',header=0,sep=';')


# In[38]:


from dataprep.eda import plot, plot_correlation, create_report, plot_missing
plot(data)


# In[21]:


plt.hist(filtered_df['Year_Birth'])


# In[22]:


plt.hist(filtered_df['Education'])


# In[26]:


plt.hist(filtered_df['Income'])


# In[24]:


plt.hist(filtered_df['MntWines'])

