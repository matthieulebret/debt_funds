import streamlit as st
import altair as alt
import seaborn as sns

import plotly.express as px

import numpy as np
import pandas as pd

from timeit import default_timer as timer
import calendar
from datetime import date, timedelta,time,datetime
import xlrd
import openpyxl

from iteration_utilities import deepflatten

from kmodes.kmodes import KModes

st.set_page_config('Infra debt funds',layout='wide')


st.title('Infrastructure debt fund analysis')

yearmin = 2001
yearmax = 2022

yearfilter = st.slider('Select period',min_value=int(yearmin),max_value=int(yearmax),value=(2017,int(yearmax)),step=1)


st.header('Debt Fund analysis by region')

@st.cache(allow_output_mutation=True)
def getdf():
    regions = ['North America','Latin America','Asia-Pacific','Western Europe','Central/Eastern Europe','Middle East/Africa']
    df = pd.read_excel('Funds_debt.xlsx',usecols=range(11),sheet_name='Funds')

    for region in regions:
        df[region] = df['Region focus'].apply(lambda x: 1 if region in x else 0)

    df = df[(df['Final close year']>=yearfilter[0])&(df['Final close year']<=yearfilter[1])]

    regcol = regions
    regcol.insert(0,'Final close year')

    minidf = df[[col for col in regcol]]
    regionheat = pd.pivot_table(minidf,columns='Final close year',aggfunc='sum')
    regionheat = pd.DataFrame(regionheat)

    return df,minidf,regionheat,regcol

df,minidf,regionheat,regcol = getdf()

st.subheader('Funds by year and region')

cm = sns.light_palette('green',as_cmap=True)
st.write(regionheat.style.background_gradient(cmap=cm,axis=None))


st.subheader('Funds by asset manager and region')


regcol = ['Institution name','Fund name'] + regcol
minidf = df[[col for col in regcol]]

anadf = minidf.iloc[:,3:]

with st.expander('Show cluster number optimisation'):
    # Choosing optimal K
    cost = []
    for cluster in range(1,12):
        try:
            kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
            kmodes.fit_predict(anadf)
            cost.append(kmodes.cost_)
        except:
            pass
    df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})

    fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
    st.plotly_chart(fig)

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

selectchart = st.selectbox('Select chart',['Institution > Segment > Fund Name','Segment > Institution > Fund Name'],1)

if selectchart == 'Institution > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Institution name','Segment','Fund name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Institution name','Fund name'],color='Institution name')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf


################################################
############### Limited Partners ###############
################################################

st.header('Debt Fund analysis by sector')

def getuniquelist(col):
    mylist = df[col].unique().tolist()
    mylist = [str(item).split(' / ') if item is not None else None for item in mylist]
    mylist = list(deepflatten(mylist,depth=1))
    mylist = list(dict.fromkeys(mylist))
    return mylist


@st.cache(allow_output_mutation=True)
def getsectordf():
    df = pd.read_excel('LP_debt.xlsx',sheet_name='Fund Commitments')
    df = df[(df['Vintage Year']>=yearfilter[0])&(df['Vintage Year']<=yearfilter[1])]
    df = df[df['Strategies']=='Debt']
    df = df.drop_duplicates(subset=['Fund Name'],keep='first',inplace=False)
    return df

df = getsectordf()

regions = getuniquelist('Regions')

fullregionlist = regions
fullregionlist.insert(0,'All')

selectregion = st.selectbox('Select region',fullregionlist)

if selectregion != 'All':
    df = df[df['Regions'].str.contains(selectregion)]

sectors = getuniquelist('Sectors')

@st.cache(allow_output_mutation=True)
def getsectorheatmap():
    for sector in sectors:
        df[sector] = df['Sectors'].apply(lambda x: 1 if sector in str(x) else 0)

    dfcol = ['Fund Name','Manager','Vintage Year']+sectors

    minidf = df[[col for col in dfcol]]
    sectorheat = pd.pivot_table(minidf,columns='Vintage Year',aggfunc='sum')
    sectorheat = pd.DataFrame(sectorheat)

    return df,minidf,sectorheat

df,minidf,sectorheat=getsectorheatmap()

st.subheader('Funds by year and sector')


cm = sns.light_palette('green',as_cmap=True)
st.write(sectorheat.style.background_gradient(cmap=cm,axis=None))


anadf = minidf.iloc[:,3:]

with st.expander('Show cluster number optimisation'):
    # Choosing optimal K
    cost = []
    for cluster in range(1,12):
        try:
            kmodes = KModes(n_jobs = -1, n_clusters = cluster, init='Huang',random_state=0)
            kmodes.fit_predict(anadf)
            cost.append(kmodes.cost_)
        except:
            pass
    df_cost = pd.DataFrame({'Cluster':range(1,12),'Cost':cost})

    fig = px.line(df_cost,x='Cluster',y='Cost',title='Optimal number of clusters')
    st.plotly_chart(fig)

k = st.number_input('Please input desired number of clusters',2,12,2,key=1)
st.warning('If you get an error message, reduce the desired number of clusters.')

k = int(k)

kmodes = KModes(n_jobs=-1,n_clusters=k,init='Huang',random_state=0)
kmodes.fit_predict(anadf)
output = pd.DataFrame(kmodes.cluster_centroids_)
output.columns = anadf.columns
output.index = ['Segment '+str(i) for i in range(k)]

output.iloc[:,:-2]


minidf['Cluster Labels']=kmodes.labels_
minidf['Segment'] = minidf['Cluster Labels'].map({i:'Segment '+str(i) for i in range(k)})

selectchart = st.selectbox('Select chart',['Manager > Segment > Fund Name','Segment > Manager > Fund Name'],1)

if selectchart == 'Manager > Segment > Fund Name':
    fig = px.treemap(minidf,path=['Manager','Segment','Fund Name'],color='Segment')
else:
    fig = px.treemap(minidf,path=['Segment','Manager','Fund Name'],color='Manager')
st.plotly_chart(fig,use_container_width=True)

with st.expander('Show data'):
    minidf



################################################
############### General Partners ###############
################################################
#
#
# st.header('General Partners')
#
# df = pd.read_excel('GP_debt.xlsx',sheet_name='Institutions')
#
# def getuniquelist(col):
#     mylist = df[col].unique().tolist()
#     mylist = [str(item).split(';') if item is not None else None for item in mylist]
#     mylist = list(deepflatten(mylist,depth=1))
#     mylist = list(dict.fromkeys(mylist))
#     return mylist
#
# strategies = getuniquelist('Allocation-Strategy')
# regions = getuniquelist('Allocation-Region')
# sectors = getuniquelist('Allocation-Sector')
