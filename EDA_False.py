#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle as pkl


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

import plotly 
import plotly.offline as py 
import plotly.graph_objs as go
import plotly.tools as tls  

from sklearn import preprocessing
from d2m.checker import duplicateCheck, VCSummary, VCCheck


# In[3]:


cateMap = {'ExistingAccountStatus':{'A11' : '... < 0 DM', 
                                    'A12' : '0 <= ... < 200 DM', 
                                    'A13' : '... >= 200 DM / salary assignments for at least 1 year', 
                                    'A14' : 'no checking account'
                                   }, 
           'CreditHistory':{'A30' : 'no credits taken/ all credits paid back duly', 
                            'A31' : 'all credits at this bank paid back duly', 
                            'A32' : 'existing credits paid back duly till now', 
                            'A33' : 'delay in paying off in the past', 
                            'A34' : 'critical account/ other credits existing (not at this bank)'}, 
           'Purpose':{'A40' : 'car (new)', 
                      'A41' : 'car (used)', 
                      'A42' : 'furniture/equipment', 
                      'A43' : 'radio/television', 
                      'A44' : 'domestic appliances', 
                      'A45' : 'repairs', 
                      'A46' : 'education', 
                      'A47' : '(vacation - does not exist?)', 
                      'A48' : 'retraining', 
                      'A49' : 'business', 
                      'A410' :' others'}, 
           'SavingAcount':{'A61' : '... < 100 DM', 
                           'A62' : '100 <= ... < 500 DM', 
                           'A63' : '500 <= ... < 1000 DM', 
                           'A64' : '.. >= 1000 DM', 
                           'A65' : 'unknown/ no savings account'}, 
           'PresentEmployment':{'A71' : 'unemployed', 
                                'A72' : '... < 1 year', 
                                'A73' : '1 <= ... < 4 years', 
                                'A74' : '4 <= ... < 7 years', 
                                'A75' : '.. >= 7 years'}, 
           'Sex':{'A91' : 'male : divorced/separated', 
                  'A92' : 'female : divorced/separated/married', 
                  'A93' : 'male : single', 
                  'A94' : 'male : married/widowed', 
                  'A95' : 'female : single'},
           'Guarantors':{'A101' : 'none',
                         'A102' : 'co-applicant',
                         'A103' : 'guarantor'}, 
           'Property':{'A121' : 'real estate', 
                       'A122' : 'if not A121 : building society savings agreement/ life insurance', 
                       'A123' : 'if not A121/A122 : car or other, not in attribute 6', 
                       'A124' : 'unknown / no property'}, 
           'Installment':{'A141' : 'bank', 
                          'A142' : 'stores', 
                          'A143' : 'none'},
           'Housing':{'A151':'rent', 
                      'A152':'own', 
                      'A153':'for free'},
           'Job':{'A171' : 'unemployed/ unskilled - non-resident', 
                  'A172' : 'unskilled - resident', 
                  'A173' : 'skilled employee / official', 
                  'A174' : 'management/ self-employed/highly qualified employee/ officer'},
           'Telephone':{'A191' : 'none', 
                        'A192' : 'yes, registered under the customers name'},
           'ForeignWorker':{'A201' : 'yes', 
                            'A202' : 'no'}}
           
def catMap(series, simple=False):
    name = series.name
    print(name)
    if simple: return [ cateMap[name][_] for _ in series] 
    else: return [ cateMap[name][_] for _ in series.value_counts().index.values] 
    
def unusedCat(df, col):
    unusedKey = set(cateMap[col]) - set(pd.unique(df[col]))
    return [cateMap[col][k] for k in unusedKey]

def vilinPlot(dfGood, dfBad, colx, coly):
    fig = {
        "data": [
            {
                "type": 'violin',
                "x": dfGood[colx],
                "y": dfGood[coly],
                "legendgroup": 'Good Credit',
                "scalegroup": 'No',
                "name": 'Good Credit',
                "side": 'negative',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'blue'
                }
            },
            {
                "type": 'violin',
                "x": dfBad[colx],
                "y": dfBad[coly],
                "legendgroup": 'Bad Credit',
                "scalegroup": 'No',
                "name": 'Bad Credit',
                "side": 'positive',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'red'
                }
            }
        ],
        "layout" : {
            "yaxis": {
                "zeroline": False,
            },
            "violingap": -1,
            "violinmode": "overlay"
        }
    }


    py.iplot(fig, filename = 'violin/split', validate = False)    
    
def barChart(dfGood, dfBad, colx):
    T1 = go.Bar(x = dfGood[colx].value_counts().index.values, y = dfGood[colx].value_counts().values, name='Good credit')
    T2 = go.Bar(x = dfBad[colx].value_counts().index.values, y = dfBad[colx].value_counts().values, name="Bad Credit")

    data = [T1, T2]

    layout = go.Layout(title=f'{colx} Distribuition')
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=f'{colx} Grouped')
    
def boxPlot(dfGood, dfBad, colx, coly):
    T1 = go.Box(y=dfGood[coly], x=dfGood[colx], name='Good credit')
    T2 = go.Box(y=dfBad[coly], x=dfBad[colx], name='Bad credit')

    data = [T1, T2]
    layout = go.Layout(yaxis=dict(title=f'{coly}',zeroline=False ), xaxis=dict(title=f'{colx} Categories' ),boxmode='group')
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename=f'{colx} Categories')


# In[4]:


def vilinPlot_(df1, df2, df3, df4, colx, coly):
    fig = {
        "data": [
            {
                "type": 'violin',
                "x": df1[colx],
                "y": df1[coly],
                "legendgroup": 'male : divorced/separated',
                "scalegroup": 'No',
                "name": 'male : divorced/separated',
                "side": 'negative',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'blue'
                }
            },
            {
                "type": 'violin',
                "x": df2[colx],
                "y": df2[coly],
                "legendgroup": 'female : divorced/separated/married',
                "scalegroup": 'No',
                "name": 'female : divorced/separated/married',
                "side": 'positive',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'red'
                }
            },
            {
                "type": 'violin',
                "x": df3[colx],
                "y": df3[coly],
                "legendgroup": 'male : single',
                "scalegroup": 'No',
                "name": 'male : single',
                "side": 'negative',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'green'
                }
            },
            {
                "type": 'violin',
                "x": df4[colx],
                "y": df4[coly],
                "legendgroup": 'male : married/widowed',
                "scalegroup": 'No',
                "name": 'male : married/widowed',
                "side": 'positive',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'yellow'
                }
            }
        ],
        "layout" : {
            "yaxis": {
                "zeroline": False,
            },
            "violingap": -1,
            "violinmode": "overlay"
        }
    }


    py.iplot(fig, filename = 'violin/split', validate = False)    
    
def barChart_(df1, df2, df3, df4, colx):
    T1 = go.Bar(x = df1[colx].value_counts().index.values, y = df1[colx].value_counts().values, name='male : divorced/separated')
    T2 = go.Bar(x = df2[colx].value_counts().index.values, y = df2[colx].value_counts().values, name='female : divorced/separated/married')
    T3 = go.Bar(x = df3[colx].value_counts().index.values, y = df3[colx].value_counts().values, name='male : single')
    T4 = go.Bar(x = df4[colx].value_counts().index.values, y = df4[colx].value_counts().values, name='male : married/widowed')


    data = [T1, T2, T3, T4]

    layout = go.Layout(title=f'{colx} Distribuition')
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=f'{colx} Grouped')
    
def boxPlot_(df1, df2, df3, df4, colx, coly):
    T1 = go.Box(y=df1[coly], x=df1[colx], name='male : divorced/separated')
    T2 = go.Box(y=df2[coly], x=df2[colx], name='female : divorced/separated/married')
    T3 = go.Box(y=df3[coly], x=df3[colx], name='male : single')
    T4 = go.Box(y=df4[coly], x=df4[colx], name='male : married/widowed')


    data = [T1, T2, T3, T4]
    layout = go.Layout(yaxis=dict(title=f'{coly}',zeroline=False ), xaxis=dict(title=f'{colx} Categories' ),boxmode='group')
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename=f'{colx} Categories')


# # 1. 資料檢視

# In[5]:


df = pd.read_csv('german.data',header = None, sep = ' ')
df.columns = ['ExistingAccountStatus','MonthDuration','CreditHistory','Purpose','CreditAmount','SavingAcount','PresentEmployment','InstalmentRate','Sex','Guarantors','Residence','Property','Age','Installment','Housing','ExistingCredits','Job','NumPeople','Telephone','ForeignWorker','Status']
categoricalFeatures=['ExistingAccountStatus','CreditHistory','Purpose','SavingAcount', 'PresentEmployment', 'Sex','Guarantors','Property','Installment','Housing','Job','Telephone','ForeignWorker']
# A8: Installment rate in percentage of disposable income = 分期付款佔可支配收入的百分比
#df = pkl.load(open('X_false_detail.pkl', 'rb'))
#df.reset_index(drop=True)


# ## 1-1. 資料集摘要

# In[6]:


df.describe()


# In[7]:


duplicateCheck(df)
duplicateCheck(df.drop(['Status'], axis=1))
duplicateCheck(df, axis=1)


# In[8]:


df.info()


# ## 1-2. 類別型特徵檢視

# In[9]:


pd.concat([pd.DataFrame(df.nunique()).loc[categoricalFeatures,].T.rename({0:'observed'}), 
 pd.DataFrame(dict((k, [len(v)]) for k, v in cateMap.items())
             ).rename({0:'# of cat.'})
]).T


# In[10]:


print(f'unused Purpose:{unusedCat(df, "Purpose")}')
print(f'unused Sex:{unusedCat(df, "Sex")}')


# In[11]:


dfEncode = df.copy()
dfDetail = df.copy()

df.head(5)


# In[13]:


for x in categoricalFeatures:
    dfDetail[x] = catMap(dfDetail[x], True)
    
dfDetail.head(5)


# In[14]:


labelEncoder = preprocessing.LabelEncoder()
for x in categoricalFeatures:
    dfEncode[x] = labelEncoder.fit_transform(dfEncode[x])

dfEncode.head(5)


# In[15]:


dfEncode.info()


# #### <font color=brik>小結</font>
# 資料集沒有重複的 row / column，故沒有多對一的疑慮。  
# 類別型特徵 `Sex` 中的 `A95`(`female : single`) 沒被用到； `Purpose` 中的 `A47`(`(vacation - does not exist?)`) 沒被用到；  
# `df.info()` 提供了所有特徵的摘要。資料集沒有任何 NaN，無需做 imputation。

# # EDA

# In[16]:


T1 = go.Bar(x = ['Good Credit', 'Bad Credit'],y = dfEncode["Status"].value_counts().values)
layout = go.Layout(title='Status Distribuition')
fig = go.Figure(data=T1, layout=layout)
py.iplot(fig, filename='Status Grouped')


# #### <font color=brik>小結</font>
# `dfEncode.Status.hist(bins = 3)` 告訴我們資料集中有 700 (70%) 筆好帳(良好貸款)和 300 (30%) 筆壞帳(不良貸款)。

# ## Split dataframe

# In[17]:


# preprocess
interval = (18, 25, 35, 60, 120)
CA = ['Student', 'Young', 'Adult', 'Senior']
dfEncode['AgeCateg'] = pd.cut(dfEncode.Age, interval, labels=CA)
dfDetail['AgeCateg'] = pd.cut(dfEncode.Age, interval, labels=CA)

Good = dfEncode['Status'] == 1
Bad = dfEncode['Status'] == 2

dfGood = df[Good]
dfBad = df[Bad]

dfEnGood = dfEncode[Good]
dfEnBad = dfEncode[Bad]

dfDeGood = dfDetail[Good]
dfDeBad = dfDetail[Bad]


# In[18]:


A91 = df['Sex'] == 'A91'
A92 = df['Sex'] == 'A92'
A93 = df['Sex'] == 'A93'
A94 = df['Sex'] == 'A94'

dfDeA91 = dfDetail[A91]
dfDeA92 = dfDetail[A92]
dfDeA93 = dfDetail[A93]
dfDeA94 = dfDetail[A94]


# In[19]:


purpose1 = df.Purpose.isin(['A40', 'A41', 'A42', 'A43'])
purpose2 = df.Purpose.isin(['A44', 'A45', 'A46', 'A47'])
purpose3 = df.Purpose.isin([ 'A48', 'A49', 'A410'])

dfDeP1 = dfDetail[purpose1]
dfDeP2 = dfDetail[purpose2]
dfDeP3 = dfDetail[purpose3]

dfDeGoodP1 = dfDetail[purpose1 & Good]
dfDeGoodP2 = dfDetail[purpose2 & Good]
dfDeGoodP3 = dfDetail[purpose3 & Good]

dfDeBadP1 = dfDetail[purpose1 & Bad]
dfDeBadP2 = dfDetail[purpose2 & Bad]
dfDeBadP3 = dfDetail[purpose3 & Bad]


# ## Age Feature

# In[20]:


barChart(dfDeGood, dfDeBad, 'AgeCateg')
boxPlot(dfDeGood, dfDeBad, 'AgeCateg', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'AgeCateg', 'CreditAmount')


# #### <font color=brik>小結</font>
# 將年齡分成四組：`Senior`(60 ~ 120)、`Adult`(35 ~ 60)、`Young`(25 ~ 35)、`Student`(18 ~ 25)。  
# 
# 則年齡組對信用額度分佈可以看出，對於好帳的`Q1` ~ `Q3` 的 50% 信用額度分佈相當接近，且 IQR 範圍一致，表示好帳信用額度大多落在相同範圍，有著接近的集中趨勢。壞帳的 IQR 範圍，在成人組、老年組的範圍大約是好帳的兩倍，故從分佈上來看有些扁平(分散)。<font color=red>相同組別之下的壞帳信用額度分佈都較好帳分散。</font>然而老年組的壞帳僅有 10 筆，為小樣本，故也有可能是抽到變異大的樣本。而學生組、青年組的信用額度因經濟能力較差，故能核准的信用額度大多沒那麼大，故 IQR  範圍較小。

# In[21]:


#First plot
trace0 = go.Histogram(
    x=dfDeGood['Age'].values,
    histnorm='probability',
    name='Good Credit'
)
#Second plot
trace1 = go.Histogram(
    x=dfDeBad['Age'].values,
    histnorm='probability',
    name='Bad Credit'
)
#Third plot
trace2 = go.Histogram(
    x=dfDetail['Age'].values,
    histnorm='probability',
    name='Overall Age'
)

#Creating the grid
fig = plotly.subplots.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='Age Distribuition')


# #### <font color=brik>小結</font>
# 觀察良好貸款、不良貸款的分佈，基本上與 box plot 差不多，呈現左偏分佈，以高峰皆落在 25-29 歲之間。

# ## Housing Feature

# In[22]:


barChart(dfDeGood, dfDeBad, 'Housing')
boxPlot(dfDeGood, dfDeBad, 'Housing', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'Housing', 'CreditAmount')


# #### <font color=brik>小結</font>
# 房產有三組：自有、`for free`、租屋。自有申請貸款為超過 70% 的大宗，即便單看自有的壞帳數量都比租屋者的總申請人數多。<font color=red>相同組別之下的壞帳信用額度分佈都較好帳分散。</font>自有與租屋在良好貸款、不良貸款的分佈相近，與 `for free` 的分散分佈較為不同。

# ## Gender Feature

# In[23]:


barChart(dfDeGood, dfDeBad, 'Sex')
boxPlot(dfDeGood, dfDeBad, 'Sex', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'Sex', 'CreditAmount')


# #### <font color=brik>小結</font>
# 性別與婚姻狀況分成五組，然而資料中卻缺少單身女性這組。按照資料集來看，單身男性為數量最大之群體，理論上單身女性數量上應當也不少。這可能是系統設計的錯誤，或是認列上的誤判(將單身女性歸入離婚/分居/婚姻女性這組)，若不是的話，就表示本資料集有著嚴重的抽樣誤差。然而若要按資料集詮釋的話，可能會得到德國的單身女性沒有申請貸款的風氣。
# 
# 性別上，男性申貸者佔 69% 為大宗。與前面特徵較為不同的地方在於，離婚男性的壞帳信用額度分佈較好帳分佈集中，除此組之外，其他<font color=red>相同組別之下的壞帳信用額度分佈都較好帳分散。</font>男性婚姻/者無論好壞帳都相當集中，然而這組只有 50 個樣本亦有可能是抽樣造成的。

# ## Purpose Feature

# In[24]:


barChart(dfDeGood, dfDeBad, 'Purpose')
boxPlot(dfDeGood, dfDeBad, 'Purpose', 'CreditAmount')
#vilinPlot(dfDeGood, dfDeBad, 'Purpose', 'CreditAmount')
vilinPlot(dfDeGoodP1, dfDeBadP1, 'Purpose', 'CreditAmount')
vilinPlot(dfDeGoodP2, dfDeBadP2, 'Purpose', 'CreditAmount')
vilinPlot(dfDeGoodP3, dfDeBadP3, 'Purpose', 'CreditAmount')


# In[27]:


barChart_(dfDeA91, dfDeA92, dfDeA93, dfDeA94, 'Purpose')
boxPlot_(dfDeA91, dfDeA92, dfDeA93, dfDeA94, 'Purpose', 'CreditAmount')
#vilinPlot_(dfDeA91, dfDeA92, dfDeA93, dfDeA94, 'Purpose', 'CreditAmount')
vilinPlot_(dfDeP1[A91], dfDeP1[A92], dfDeP1[A93], dfDeP1[A94], 'Purpose', 'CreditAmount')
vilinPlot_(dfDeP2[A91], dfDeP2[A92], dfDeP2[A93], dfDeP2[A94], 'Purpose', 'CreditAmount')
vilinPlot_(dfDeP3[A91], dfDeP3[A92], dfDeP3[A93], dfDeP3[A94], 'Purpose', 'CreditAmount')


# #### <font color=brik>小結</font>
# 申請目的分成 11 組，目的最多的依序是影音系統、新車車貸、傢俱/設備、二手車車貸、商業貸款、學貸、修繕、職訓、入籍申請、其他。其中職訓、入籍申請、其他的資料筆數分別為 8筆、12筆、12筆，實為小樣本。而前四大目的中影音系統、新車車貸、傢俱/設備在好壞帳的信用額度分佈都很類似，二手車車貸的信用額度分佈則跟其他人都不同。
# 
# 而商業貸款有 97 筆，約佔整體資料 10%，然而卻呈現出均勻分布，而入籍申請的好帳、修繕的壞帳所呈現的也非鐘型分佈。
# 
# 若將每個目的再依性別婚姻分類，則幾乎只有前四大目的(影音系統、新車車貸、傢俱/設備、二手車車貸)仍呈現鐘型分佈，其他目的大概是因為又被分組導致樣本太少而呈現出均勻分布的態勢。

# ## Job Feature

# In[28]:


barChart(dfDeGood, dfDeBad, 'Job')
boxPlot(dfDeGood, dfDeBad, 'Job', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'Job', 'CreditAmount')


# #### <font color=brik>小結</font>
# 職業類型分成四組：技術人員/公務員、非技術居民、經理/自僱者/高級員工/辦公人員、失業/無技術者。失業/無技術者樣本數為 22，為小樣本。經理/自僱者/高級員工/辦公人員有較高的信用額度。而人數前二多的技術人員/公務員、非技術居民其分佈較為類似。

# ## Foreign Worker Feature

# In[29]:


barChart(dfDeGood, dfDeBad, 'ForeignWorker')
boxPlot(dfDeGood, dfDeBad, 'ForeignWorker', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'ForeignWorker', 'CreditAmount')


# #### <font color=brik>小結</font>
# 工作者分為外勞/本勞，申請貸款者以外勞(`yes`)為大宗，佔總樣本數 96% 以上。外國籍的好帳、壞帳分佈接近。本勞(`no`)的好帳相當集中，壞帳則相當分散，其最大值亦為四種組合(國籍x好壞帳)中的最大者，達18.424k，然而此本勞壞帳僅有 4 例，實為小樣本，亦有可能是抽到極端值所致。<font color=red>相同組別之下的壞帳信用額度分佈都較好帳分散。</font>
# 
# 從樣本比例來看，該銀行服務對象極有可能是以外籍勞工為主，或是德國勞工沒有申請貸款的習慣。

# ## Property Feature

# In[30]:


barChart(dfDeGood, dfDeBad, 'Property')
boxPlot(dfDeGood, dfDeBad, 'Property', 'CreditAmount')
vilinPlot(dfDeGood, dfDeBad, 'Property', 'CreditAmount')


# #### <font color=brik>小結</font>
# 資產分為：不動產、(沒不動產之下)持有建房互助會存款合約或壽險、無資產、(沒不動產、沒之下建房互助會存款合約或壽險)持有動產或存款之外的財產。<font color=red>相同組別之下的壞帳信用額度分佈都較好帳分散。</font>持有不動產者不論好壞帳其信用額度分佈較為集中，而持有「建房互助會存款合約」以及持有動產者好壞帳的信用額度分佈較為相似。無資產者的信用額度就較為低闊，且無資產者的信用額度，不論好壞帳平均而言皆比持有動產者高。

# # 計算 Feature 相關係數

# In[31]:


data = dfEncode
X = data.iloc[:,:-1]  #independent columns
y = data.iloc[:,-1]    #target column
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[32]:


corrmat[corrmat!=1].Status.max(), corrmat[corrmat!=1].Status.min()


# In[33]:


corrmat.Status[~corrmat[np.abs(corrmat) > 0.1].Status.isna()]


# |相關係數範圍(絕對值)|變項關聯程度|
# |--|--|
# |1.00|完全相關| 
# |.70 至.99|高度相關| 
# |.40 至.69|中度相關| 
# |.10 至.39|低度相關|
# |.10 以下|微弱或無相關|

# #### <font color=brik>小結</font>
# 所有的特徵與貸款狀態(`Status`) 除  
# `ExistingAccountStatus`、`MonthDuration`、`CreditHistory`、`CreditAmount`、
# `SavingAcount`、`PresentEmployment`、`Property`、`Installment`  
# 為低度正負相關之外，其他特徵皆為微弱或無相關。
