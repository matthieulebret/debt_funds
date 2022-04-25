import streamlit as st
# import altair as alt
import seaborn as sns

# import requests
import plotly.express as px

import numpy as np
import pandas as pd

# import json

from timeit import default_timer as timer
import calendar
from datetime import date, timedelta,time,datetime
# import time
import xlrd
import openpyxl
# import random

# from iteration_utilities import deepflatten

from kmodes.kmodes import KModes


st.set_page_config('Infra debt funds',layout='wide')

cola,colb,colc,cold = st.columns(4)
with cola:
    st.image('https://images.unsplash.com/photo-1608237963573-ba0790bc6404?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8aW5mcmFzdHJ1Y3R1cmV8ZW58MHx8MHx8&auto=format&fit=crop&w=400&q=60')
with colb:
    st.image('https://images.unsplash.com/photo-1564957341074-6520b6b483bd?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MjF8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')
with colc:
    st.image('https://images.unsplash.com/photo-1567264837824-c993e26bf663?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mjl8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')
with cold:
    st.image('https://images.unsplash.com/photo-1482341232961-3e5973a98e7b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NDd8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')

st.title('Infrastructure debt fund analysis')

yearmin = 2001
yearmax = 2022

yearfilter = st.slider('Select period',min_value=int(yearmin),max_value=int(yearmax),value=(2017,int(yearmax)),step=1)

st.image('https://images.unsplash.com/photo-1607288531629-1e92ebd52811?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MzZ8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')

st.header('Debt Fund analysis - Global')

regions = ['North America','Latin America','Asia-Pacific','Western Europe','Central/Eastern Europe','Middle East/Africa']
strategies = ['Mezzanine / Junior','Senior debt','Fund of Funds / Co-Investment','Opportunistic']
sectors = ['Renewables, Waste, Water','Energy & Mining','Real estate','Corporates','Agri & Timberland','Infrastructure']

def getuniquelist(col):
    mylist = df[col].unique().tolist()
    mylist = [str(item).split(' / ') if item is not None else None for item in mylist]
    mylist = list(deepflatten(mylist,depth=1))
    mylist = list(dict.fromkeys(mylist))
    return mylist


@st.cache(allow_output_mutation=True)
def getdf():
    df = pd.read_excel('debt_funds_data.xlsx',sheet_name='Debt_Fund_Master_Upload').iloc[:,1:]
    df = df[(df['Vintage year']>=yearfilter[0])&(df['Vintage year']<=yearfilter[1])]

    heat1 = pd.pivot_table(df,values=regions,columns='Vintage year',aggfunc='sum')
    heat2 = pd.pivot_table(df,values=strategies,columns='Vintage year',aggfunc='sum')
    heat3 = pd.pivot_table(df,values=sectors,columns='Vintage year',aggfunc='sum')
    return df,heat1,heat2,heat3

df,heat1,heat2,heat3 = getdf()


st.subheader('Funds by year')


cm = sns.light_palette('green',as_cmap=True)
st.write(heat1.style.background_gradient(cmap=cm,axis=None))
st.write(heat2.style.background_gradient(cmap=cm,axis=None))
st.write(heat3.style.background_gradient(cmap=cm,axis=None))


st.image('https://images.unsplash.com/photo-1532418852691-7d16d021264c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NDR8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')

st.header('Debt Fund analysis - Filters')

regsel = regions
regsel.insert(0,'All')
stratsel = strategies
stratsel.insert(0,'All')
sectorsel = sectors
sectorsel.insert(0,'All')


with st.form('selection'):
    col1,col2,col3 = st.columns(3)

    with col1:
        region = st.selectbox('Select region',regsel,0)
    with col2:
        strat = st.selectbox('Select strategy',stratsel,0)
    with col3:
        sector = st.selectbox('Select sector',sectorsel,0)

    submitted = st.form_submit_button('Filter')

    if submitted:
        if region!='All':
            df = df[df[region]==1]
        if strat!='All':
            df = df[df[strat]==1]
        if sector!='All':
            df = df[df[sector]==1]

with st.expander('Show data'):
    df


st.subheader('Co-occurrence matrix')
st.warning('Note: there are several occurrences of multiple counts, making it sometimes difficult to reconcile numbers.')

@st.cache(allow_output_mutation=True)
def getmatrix():
    coocc1 = df[sectors[1:]].astype(int).T.dot(df[strategies[1:]].astype(int))
    coocc2 = df[regions[1:]].astype(int).T.dot(df[strategies[1:]].astype(int))
    coocc3 = df[regions[1:]].astype(int).T.dot(df[sectors[1:]].astype(int))
    return coocc1,coocc2,coocc3

coocc1,coocc2,coocc3 = getmatrix()

cm = sns.light_palette('green',as_cmap=True)
st.write(coocc1.style.background_gradient(cmap=cm,axis=None))
st.write(coocc2.style.background_gradient(cmap=cm,axis=None))
st.write(coocc3.style.background_gradient(cmap=cm,axis=None))



st.image('https://images.unsplash.com/photo-1542463873-d913b21db820?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NTB8fGluZnJhc3RydWN0dXJlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=400&q=60')

st.header('Segmentation - Clustering analysis')
st.warning('The filters are inherited from the other parts of the application.')

st.subheader('Segmentation by Region')

regcol = ['Institution name','Fund name'] + regions[1:]
minidf = df[[col for col in regcol]]
anadf = minidf.iloc[:,2:]

# with st.expander('Show cluster number optimisation'):
#     # Choosing optimal K
#     cost = []
#     for cluster in range(1,12):
#         try:
#             kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
#             kmodes.fit_predict(anadf)
#             cost.append(kmodes.cost_)
#         except:
#             pass
#     df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})
#
#     fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
#     st.plotly_chart(fig)

k = st.number_input('Please input desired number of clusters',2,12,4,key=0)
k = int(k)

kmodes = KModes(n_jobs=-1,n_clusters=k,init='Huang',random_state=0)
kmodes.fit_predict(anadf)
output = pd.DataFrame(kmodes.cluster_centroids_)
output.columns = anadf.columns
output.index = ['Segment '+str(i) for i in range(k)]

output

minidf['Cluster Labels']=kmodes.labels_
minidf['Segment'] = minidf['Cluster Labels'].map({i:'Segment '+str(i) for i in range(k)})

splitseg = minidf.groupby('Segment')['Cluster Labels'].count()/len(minidf)
splitseg = pd.DataFrame(splitseg).style.format('{:,.2%}')
splitseg

selectchart = st.selectbox('Select chart',['Institution > Segment > Fund Name','Segment > Institution > Fund Name'],1)

if selectchart == 'Institution > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Institution name','Segment','Fund name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Institution name','Fund name'],color='Institution name')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf


st.subheader('Segmentation by Strategy')

regcol = ['Institution name','Fund name'] + strategies[1:]
minidf = df[[col for col in regcol]]
anadf = minidf.iloc[:,2:]

# with st.expander('Show cluster number optimisation'):
#     # Choosing optimal K
#     cost = []
#     for cluster in range(1,12):
#         try:
#             kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
#             kmodes.fit_predict(anadf)
#             cost.append(kmodes.cost_)
#         except:
#             pass
#     df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})
#
#     fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
#     st.plotly_chart(fig)

k = st.number_input('Please input desired number of clusters',2,12,4,key=1)
k = int(k)

kmodes = KModes(n_jobs=-1,n_clusters=k,init='Huang',random_state=0)
kmodes.fit_predict(anadf)
output = pd.DataFrame(kmodes.cluster_centroids_)
output.columns = anadf.columns
output.index = ['Segment '+str(i) for i in range(k)]

output


minidf['Cluster Labels']=kmodes.labels_
minidf['Segment'] = minidf['Cluster Labels'].map({i:'Segment '+str(i) for i in range(k)})

splitseg = minidf.groupby('Segment')['Cluster Labels'].count()/len(minidf)
splitseg = pd.DataFrame(splitseg).style.format('{:,.2%}')
splitseg

selectchart = st.selectbox('Select chart',['Institution > Segment > Fund Name','Segment > Institution > Fund Name'],1,key=1)

if selectchart == 'Institution > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Institution name','Segment','Fund name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Institution name','Fund name'],color='Institution name')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf



st.subheader('Segmentation by Sector')

regcol = ['Institution name','Fund name'] + sectors[1:]
minidf = df[[col for col in regcol]]
anadf = minidf.iloc[:,2:]

# with st.expander('Show cluster number optimisation'):
#     # Choosing optimal K
#     cost = []
#     for cluster in range(1,12):
#         try:
#             kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
#             kmodes.fit_predict(anadf)
#             cost.append(kmodes.cost_)
#         except:
#             pass
#     df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})
#
#     fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
#     st.plotly_chart(fig)

k = st.number_input('Please input desired number of clusters',2,12,4,key=2)
k = int(k)

kmodes = KModes(n_jobs=-1,n_clusters=k,init='Huang',random_state=0)
kmodes.fit_predict(anadf)
output = pd.DataFrame(kmodes.cluster_centroids_)
output.columns = anadf.columns
output.index = ['Segment '+str(i) for i in range(k)]

output

minidf['Cluster Labels']=kmodes.labels_
minidf['Segment'] = minidf['Cluster Labels'].map({i:'Segment '+str(i) for i in range(k)})

splitseg = minidf.groupby('Segment')['Cluster Labels'].count()/len(minidf)
splitseg = pd.DataFrame(splitseg).style.format('{:,.2%}')
splitseg


selectchart = st.selectbox('Select chart',['Institution > Segment > Fund Name','Segment > Institution > Fund Name'],1,key=2)

if selectchart == 'Institution > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Institution name','Segment','Fund name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Institution name','Fund name'],color='Institution name')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf


st.subheader('Segmentation by Region, Strategy, Sector')

regcol = ['Institution name','Fund name'] + regions[1:] + strategies[1:] + sectors[1:]
minidf = df[[col for col in regcol]]
anadf = minidf.iloc[:,2:]

# with st.expander('Show cluster number optimisation'):
#     # Choosing optimal K
#     cost = []
#     for cluster in range(1,12):
#         try:
#             kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
#             kmodes.fit_predict(anadf)
#             cost.append(kmodes.cost_)
#         except:
#             pass
#     df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})
#
#     fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
#     st.plotly_chart(fig)

k = st.number_input('Please input desired number of clusters',2,12,4,key=3)
k = int(k)

kmodes = KModes(n_jobs=-1,n_clusters=k,init='Huang',random_state=0)
kmodes.fit_predict(anadf)
output = pd.DataFrame(kmodes.cluster_centroids_)
output.columns = anadf.columns
output.index = ['Segment '+str(i) for i in range(k)]

output

minidf['Cluster Labels']=kmodes.labels_
minidf['Segment'] = minidf['Cluster Labels'].map({i:'Segment '+str(i) for i in range(k)})

splitseg = minidf.groupby('Segment')['Cluster Labels'].count()/len(minidf)
splitseg = pd.DataFrame(splitseg).style.format('{:,.2%}')
splitseg

selectchart = st.selectbox('Select chart',['Institution > Segment > Fund Name','Segment > Institution > Fund Name'],1,key=3)

if selectchart == 'Institution > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Institution name','Segment','Fund name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Institution name','Fund name'],color='Institution name')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf







# API to INFRASTRUCTUREINVESTOR
# url = 'https://ra.pei.blaize.io/api/v1/funds/'
#
# placeholder = st.empty()
# progholder = st.empty()
# mybar = st.progress(0)
#
# bigdf = pd.DataFrame()
#
# fund=1
# for i in totalidlist:
#     newurl = url + str(int(i))
#     r = requests.get(newurl)
#     # with open(r,encoding='utf-8') as f:
#     #     data = f.read()
#     #     data = json.loads(data)
#     #     data = data['data']
#     df = pd.json_normalize(r.json())
#     bigdf = pd.concat([bigdf,df])
#
#     time.sleep(random.randint(1,4))
#     with placeholder:
#         st.write('File #{0} complete '.format(fund)+'/ '+str(76)+'.')
#     with progholder:
#         pct_complete = '{:,.2%}'.format(fund/76)
#         st.write(pct_complete,' complete.' )
#         try:
#             mybar.progress(fund/76)
#         except:
#             mybar.progress(1)
#     fund=fund+1
#
