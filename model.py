#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from d2m.checker import duplicateCheck, VCSummary, VCCheck

import plotly
import plotly.offline as py
import plotly.express as px
from plotly.graph_objs import Figure
import pickle as pkl
import random


# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from packaging import version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."


from numpy.random import seed
random_seed = 2021
seed(random_seed)
tf.random.set_seed(random_seed)


# In[3]:


def analize(y_pred, y_true, modelName='', n_classes = 2):
    # ROC, AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    trace1 = {
      'name': f'Status = good, auc={auc(fpr[0], tpr[0]):.2f}', 
      'type': 'scatter', 
      'x': fpr[0], 
      'y': tpr[0]
    }
    trace2 = {
      'name': f'Status = bad, auc={auc(fpr[1], tpr[1]):.2f}', 
      'type': 'scatter', 
      'x': fpr[1], 
      'y': tpr[1]
    }

    data = [trace1, trace2]
    layout = {
      'shapes': [{'line': {'dash': 'dash'}, 'type': 'line', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}],
      'title': f'ROC curve {modelName}', 
      'xaxis': {
        'title': 'false positive rate', 
        'zeroline': False
      }, 
      'yaxis': {
        'title': 'true positive rate', 
        'zeroline': False,
      },
      'width':650,
      'height':500,
    }

    fig = Figure(data=data, layout=layout)
    py.iplot(fig, validate = False)
    
    # confusion matrix
    y_pred_cat = np.argmax(y_pred.round(), axis = 1)
    y_true_cat = np.argmax(y_true, axis = 1)
    print(f'accuracy_score: {accuracy_score(y_true_cat, y_pred_cat)}')

    cm = pd.DataFrame(confusion_matrix(y_pred_cat, y_true_cat))
    cm.columns = ['true 0', 'true 1']
    cm.index = ['pred 0', 'pred 1']
    
    TP, FP = cm.iloc[0,0], cm.iloc[0,1]
    FN, TN = cm.iloc[1,0], cm.iloc[1,1]
    
    
    sensitivity = TP/(TP+FN)
    recall = sensitivity
    specificity = TN/(FP+TN)
    precision = TP/(TP+FP)    
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    f1score = 2/(1/precision+1/recall)
    
    summary = pd.DataFrame({'metric':['Sensitivity', 'Recall', 'Specificity', 'Precision', 'Accuracy', 'F1Score'], 
                            'value':[sensitivity, recall, specificity, precision, accuracy, f1score]})
    return cm, summary


# In[4]:


def plotHistory(history, col):
    trace1 = {
      'name': f'{col}', 
      'type': 'scatter', 
      'x': list(range(len(history[f'{col}']))), 
      'y': history[f'{col}']
    }
    trace2 = {
      'name': f'val_{col}', 
      'type': 'scatter', 
      'x': list(range(len(history[f'val_{col}']))), 
      'y': history[f'val_{col}']
    }

    data = [trace1, trace2]
    layout = {
      'title': f'history of {col} vs vol_{col}', 
      'xaxis': {
        'title': 'false positive rate', 
        'zeroline': False
      }, 
      'yaxis': {
        'title': 'true positive rate', 
        'zeroline': False,
      },
    }

    fig = Figure(data=data, layout=layout)
    py.iplot(fig, validate = False)
    
def printHistory(history):
    h = history
    print(f"loss:{h['loss'][-1]:.4e} accuracy:{h['accuracy'][-1]:.4e} val_loss:{h['val_loss'][-1]:.4e} val_accuracy:{h['val_accuracy'][-1]:.4e}")


# In[5]:


df = pd.read_csv('german.data',header = None, sep = ' ')
df.columns = ['ExistingAccountStatus','MonthDuration','CreditHistory','Purpose','CreditAmount','SavingAcount','PresentEmployment','InstalmentRate','Sex','Guarantors','Residence','Property','Age','Installment','Housing','ExistingCredits','Job','NumPeople','Telephone','ForeignWorker','Status']
categoricalFeatures=['ExistingAccountStatus','CreditHistory','Purpose','SavingAcount', 'PresentEmployment', 'Sex','Guarantors','Property','Installment','Housing','Job','Telephone','ForeignWorker']

dfEncode = df.copy()
dfDetail = df.copy()

df.head(5)


# In[6]:


labelEncoder = preprocessing.LabelEncoder()
for x in categoricalFeatures:
    dfEncode[x] = labelEncoder.fit_transform(dfEncode[x])

dfEncode['Status'] = labelEncoder.fit_transform(dfEncode['Status']) # {1:0, 2:1}
onehotStatus = np_utils.to_categorical(dfEncode['Status'])
dfEncode.head(5)


# # 檢查資料量是否足夠

# In[7]:


VCSummary(dfEncode, dvc = 100, epsilon=0.05, delta=0.05)


# #### <font color=brik>小結</font>
# 對模型參數量 > 100 的 model，目前 1000 筆的 dataset 並不足以支撐訓練這樣的模型。故先選擇 NN 作為 feature transform 後的分類器當作 baselien。接下來選擇以較容易 overfitting 的模型 random-forest 以及 SVC 做為分類器，最終再 ensemble 這兩個 model 以盡量消減 model bias。

# # Check on PCA

# In[8]:


from sklearn.decomposition import PCA
X = dfEncode.drop(['Status'], axis=1)
#y =np_utils.to_categorical(dfEncode.Status)
pca = PCA(n_components=8)
pca.fit(X)
components = pca.fit_transform(X)
components = pd.DataFrame(components[:,0:2], columns=['PCA0', 'PCA1'])
components.loc[:,'Status'] = dfEncode.Status
components.Status = components.Status.apply(lambda x: 'Bad' if x  else 'Good')


# In[9]:


df = px.data.iris() # iris is a pandas DataFrame
fig = px.scatter(components, x='PCA0', y='PCA1', color='Status', title='PCA of Dataset')
fig.show()


# #### <font color=brik>小結</font>
# 做完 PCA 發現 Status 的 Good、Bad 還是有一堆擠在一起，表示即使變異最大的兩軸仍難以分開，因此肯定需要做高維度的 feature transform 才有可能分得好。

# # 實驗紀錄宣告

# In[10]:


test_summary = dict()
train_summary = dict()


# ## DNN 分類器

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(dfEncode.drop(['Status'], axis=1), onehotStatus, 
                                                    test_size=0.2, 
                                                    random_state=random_seed)

X_train_detail = dfDetail.iloc[X_train.index,]
X_test_detail = dfDetail.iloc[X_test.index,]

scaler = preprocessing.StandardScaler().fit(X_train)
scaler.mean_, scaler.scale_

X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)


# In[12]:


def NN(dim, mute=True):
    input = tf.keras.Input(shape=(20, ), name='input')
    DNN1 = keras.layers.Dense(16, activation='relu', name='dnn1')(input)
    DNN2 = keras.layers.Dense(8, activation='relu', name='dnn2')(DNN1)
    #DNN3 = keras.layers.Dense(5, activation='softmax', name='dnn3')(DNN2)
    output = keras.layers.Dense(2, activation='softmax', name='output')(DNN2)
    
    model = keras.Model(input, output)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    if not mute: print(model.summary())
    
    return model


# In[13]:


dnn = NN(20, False)


# In[14]:


tf.keras.utils.plot_model(dnn, './dnn.png')


# In[15]:


dnnHistory = dnn.fit(X_train_, y_train, 
                     validation_data=(X_test_, y_test),
                     batch_size=4, 
                     epochs=400, 
                     verbose=False)
printHistory(dnnHistory.history)


# In[16]:


plotHistory(dnnHistory.history, 'accuracy')
plotHistory(dnnHistory.history, 'loss')


# In[17]:


y_pred = dnn.predict(X_test_)#.round()
cm, summary = analize(y_pred, y_test, 'of baseline model DNN')
test_summary['dnn'] = summary.copy()


# In[18]:


cm


# In[19]:


summary


# In[20]:


y_pred = dnn.predict(X_train_)#.round()
cm, summary = analize(y_pred, y_train, 'of baseline model DNN')
train_summary['dnn'] = summary.copy()


# In[21]:


cm


# In[22]:


summary


# # SVC

# In[23]:


from sklearn.svm import SVC
svc = SVC(gamma='auto', class_weight = 'balanced')
svc.fit(X_train_, np.argmax(y_train, axis = 1))

y_pred = np_utils.to_categorical(svc.predict(X_test_))
cm, summary = analize(y_pred, y_test, 'of SVC')
test_summary['svc'] = summary.copy()


# In[24]:


cm


# In[25]:


summary


# In[26]:


y_pred = np_utils.to_categorical(svc.predict(X_train_))
cm, summary = analize(y_pred, y_train, 'of SVC')
train_summary['svc'] = summary.copy()


# In[27]:


cm


# In[28]:


summary


# # Random forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1234)
randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_test)
cm, summary = analize(y_pred, y_test, 'of Random Forest')
test_summary['randomforest'] = summary.copy()


# In[30]:


cm


# In[31]:


summary


# In[32]:


y_pred = randomforest.predict(X_train)
cm, summary = analize(y_pred, y_train, 'of Random Forest')
train_summary['randomforest'] = summary.copy()


# In[33]:


cm


# In[34]:


summary


# # XGBoost 

# In[35]:


from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_train, np.argmax(y_train, axis = 1))

y_pred = np_utils.to_categorical(xgb.predict(X_test))
cm, summary = analize(y_pred, y_test, 'of XGBoost')
test_summary['xgb'] = summary.copy()


# In[36]:


cm


# In[37]:


summary


# In[38]:


y_pred = np_utils.to_categorical(xgb.predict(X_train))
cm, summary = analize(y_pred, y_train, 'of XGBoost')
train_summary['xgb'] = summary.copy()


# In[39]:


cm


# In[40]:


summary


# # Decision Tree

# In[41]:


from sklearn import tree
decisiontree = tree.DecisionTreeClassifier()
decisiontree = decisiontree.fit(X_train, y_train)

y_pred = decisiontree.predict(X_test)
cm, summary = analize(y_pred, y_test, 'of Decision Tree')
test_summary['decisiontree'] = summary.copy()


# In[42]:


cm


# In[43]:


summary


# In[44]:


y_pred = decisiontree.predict(X_train)
cm, summary = analize(y_pred, y_train, 'of Decision Tree')
train_summary['decisiontree'] = summary.copy()


# In[45]:


cm


# In[46]:


summary


# # 各種 model 在 testing set 上的比較

# In[ ]:





# # Ensemble

# In[47]:


def ensemble(models, Xs, onehots):
    y_pred = None
    i = 0
    for ((name, model), X, onehot) in zip(models.items(), Xs, onehots):
        print(name, onehot)
        if y_pred is None:
            pred = model.predict(X)
            y_pred = np_utils.to_categorical(pred) if onehot else pred
        else:   
            pred = model.predict(X)
            y_pred += np_utils.to_categorical(pred) if onehot else pred
    y_pred = y_pred/len(models)
    return y_pred


# In[48]:


y_pred = ensemble({'dnn':dnn, 
                   'svc':svc}, 
                  [X_test_, X_test_], 
                  [False, True])
cm, summary = analize(y_pred, y_test, 'of ensemble')
test_summary['dnn+svc'] = summary.copy()


# In[49]:


cm


# In[50]:


summary


# In[51]:


y_pred = ensemble({'dnn':dnn, 
                   'svc':svc, 
                   'rforest':randomforest}, 
                  [X_test_, X_test_, X_test], 
                  [False, True, False])
cm, summary = analize(y_pred, y_test, 'of ensemble')
test_summary['dnn+svc+rforest'] = summary.copy()


# In[52]:


cm


# In[53]:


summary


# In[54]:


y_pred = ensemble({'dnn':dnn, 
                   'svc':svc, 
                   'rforest':randomforest, 
                   'dtree':decisiontree}, 
                  [X_test_, X_test_, X_test, X_test], 
                  [False, True, False, False])
cm, summary = analize(y_pred, y_test, 'of ensemble')
test_summary['dnn+svc+rforest+dtree'] = summary.copy()


# In[55]:


cm


# In[56]:


summary


# In[57]:


y_pred = ensemble({'dnn':dnn, 
                   'svc':svc, 
                   'rforest':randomforest, 
                   'xgb':xgb,
                   'dtree':decisiontree}, 
                  [X_test_, X_test_, X_test, X_test, X_test], 
                  [False, True, False, True, False])
cm, summary = analize(y_pred, y_test, 'of ensemble')
test_summary['dnn/svc/rforest/dtree/xgb'] = summary.copy()


# In[58]:


cm


# In[59]:


summary


# # Check test summary

# In[60]:


pkl.dump(test_summary, open('test_summary.pkl', 'wb'))


# # Check false class 0 samples

# In[61]:


ys = pd.DataFrame({'pred':np.argmax(y_pred, axis=1), 'true':np.argmax(y_test, axis=1)})
y_false0 = ys[(ys.pred!=ys.true) & (ys.pred==0)]
y_false1 = ys[(ys.pred!=ys.true) & (ys.pred==1)]

X_false0 = X_test.reset_index(drop=True).iloc[y_false0.index,:]
X_false1 = X_test.reset_index(drop=True).iloc[y_false1.index,:]

X_false0['Status'] = 2
X_false1['Status'] = 1

X_false = pd.concat([X_false0, X_false1])


X_false0_detail = X_test_detail.reset_index(drop=True).iloc[y_false0.index,:]
X_false1_detail = X_test_detail.reset_index(drop=True).iloc[y_false1.index,:]

X_false0_detail['Status'] = 2
X_false1_detail['Status'] = 1

X_false_detail = pd.concat([X_false0_detail, X_false1_detail])


# In[62]:


pkl.dump(X_false, open('X_false.pkl', 'wb'))
pkl.dump(X_false_detail, open('X_false_detail.pkl', 'wb'))


# # Bagging

# In[63]:


def bagging(X_train, y_train, X_test, y_test, n=10, models=['dnn', 'svc', 'rforest', 'xgb', 'dtree'], frac=0.75):
    df = X_train.copy()
    df[0] = y_train[:,0]
    df[1] = y_train[:,1]
    
    ys = []
    #onehots = []
    dnnInfo = []
    while (n :=n-1) >= 0:
        X_train = df.sample(frac=frac)
        y_train = X_train[[0, 1]].values
        X_train = X_train.drop([0, 1], axis=1)
        X_train_ = scaler.transform(X_train)

        for model in models:
            if model == 'dnn':
                tf.random.set_seed(random_seed+random.randint(1,1000000))
                m = NN(20)
                
                mHistory = m.fit(X_train_, y_train, 
                                 validation_data=(X_test_, y_test),
                                 batch_size=4, 
                                 epochs=int(400*frac), 
                                 verbose=False)
                dnnInfo.append(pd.DataFrame({'loss':mHistory.history['loss'], 
                                             'accuracy':mHistory.history['accuracy'], 
                                             'val_loss':mHistory.history['val_loss'], 
                                             'val_accuracy':mHistory.history['val_accuracy']}))
                ys.append(m.predict(X_test_))
                
            elif model == 'svc':
                m = SVC(gamma='auto', degree=3)
                m.fit(X_train_, np.argmax(y_train, axis = 1))
                ys.append(np_utils.to_categorical(m.predict(X_test_)))

            elif model == 'rforest':
                m = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1234)
                m.fit(X_train, y_train)
                ys.append(m.predict(X_test))

            elif model == 'xgb':
                m = XGBClassifier(use_label_encoder=False)
                m.fit(X_train, np.argmax(y_train, axis = 1))
                ys.append(np_utils.to_categorical(m.predict(X_test)))
                
            elif model == 'dtree':
                m = tree.DecisionTreeClassifier()
                m.fit(X_train, y_train)
                ys.append(m.predict(X_test))

    return ys, dnnInfo

def avgYs(ys):
    y_pred = None
    for y in ys:
        if y_pred is None:
            y_pred = y
        else:
            y_pred += y
    else:
        y_pred = y_pred/len(ys)
    return y_pred


# In[64]:


bagging_experiment = dict()


# In[65]:


bagging_experiment['origin'] = {'cm':cm.copy(), 'summary':summary.copy()}


# # Baggin of 5 models per algo., frac = 1

# In[66]:


n = 5
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[67]:


cm


# In[68]:


summary


# # Baggin of 10 models per algo., frac = 1

# In[69]:


n = 10
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[70]:


cm


# In[71]:


summary


# # Baggin of 15 models per algo., frac = 1

# In[72]:


n = 15
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[73]:


cm


# In[74]:


summary


# # Baggin of 20 models per algo., frac = 1

# In[75]:


n = 20
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[76]:


cm


# In[77]:


summary


# # Baggin of 25 models per algo., frac = 1

# In[78]:


n = 25
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[79]:


cm


# In[80]:


summary


# # Baggin of 30 models per algo., frac = 1

# In[81]:


n = 30
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=1)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=1'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[82]:


cm


# In[83]:


summary


# # Baggin of 5 models per algo., frac = 0.75

# In[84]:


n = 5
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[85]:


cm


# In[86]:


summary


# # Bagging of 10 models per algo. frac = 0.75

# In[87]:


n=10
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[88]:


cm


# In[89]:


summary


# # Bagging of 15 models per algo. frac = 0.75

# In[90]:


n=15
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[91]:


cm


# In[92]:


summary


# # Bagging of 20 models per algo. frac = 0.75

# In[93]:


n=20
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[94]:


cm


# In[95]:


summary


# # Bagging of 25 models per algo. frac = 0.75

# In[96]:


n=25
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[97]:


cm


# In[98]:


summary


# # Bagging of 30 models per algo. frac = 0.75

# In[99]:


n = 30
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.75'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[100]:


cm


# In[101]:


summary


# # Baggin of 5 models per algo., frac = 0.5

# In[102]:


n = 5
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[103]:


cm


# In[104]:


summary


# # Baggin of 10 models per algo., frac = 0.5

# In[105]:


n = 10
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[106]:


cm


# In[107]:


summary


# # Baggin of 15 models per algo., frac = 0.5

# In[108]:


n = 15
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[109]:


cm


# In[110]:


summary


# # Baggin of 20 models per algo., frac = 0.5

# In[111]:


n = 20
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[112]:


cm


# In[113]:


summary


# # Baggin of 25 models per algo., frac = 0.5

# In[114]:


n = 25
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[115]:


cm


# In[116]:


summary


# # Baggin of 30 models per algo., frac = 0.5

# In[117]:


n = 30
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.5)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.5'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[118]:


cm


# In[119]:


summary


# # Baggin of 5 models per algo., frac = 0.25

# In[120]:


n = 5
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[121]:


cm


# In[122]:


summary


# # Baggin of 10 models per algo., frac = 0.25

# In[123]:


n = 10
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[124]:


cm


# In[125]:


summary


# # Baggin of 15 models per algo., frac = 0.25

# In[126]:


n = 15
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[127]:


cm


# In[128]:


summary


# # Baggin of 20 models per algo., frac = 0.25

# In[129]:


n = 20
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[130]:


cm


# In[131]:


summary


# # Baggin of 25 models per algo., frac = 0.25

# In[132]:


n = 25
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[133]:


cm


# In[134]:


summary


# # Baggin of 30 models per algo., frac = 0.25

# In[135]:


n = 50
ys, dnnInfo = bagging(X_train, y_train, X_test, y_test, n=n, frac=0.25)
y_pred = avgYs(ys)
cm, summary = analize(y_pred, y_test, f'bag of {n} models per algorithm')
bagging_experiment[f'n={n}, frac=0.25'] = {'cm':cm.copy(), 'summary':summary.copy()}


# In[136]:


cm


# In[137]:


summary


# # Save experiment result

# In[139]:


pkl.dump(bagging_experiment, open('baggingExp.pkl', 'wb'))


# # Bagging Result

# In[140]:


# e = bagging_experiment
# e.keys()


# In[141]:


# e['origin']['cm']


# In[142]:



# experiment = {'origin':[1, 100,
#                         e['origin']['cm'].loc['pred 0','true 0'], 
#                         e['origin']['cm'].loc['pred 0','true 1'], 
#                         e['origin']['cm'].loc['pred 1','true 0'], 
#                         e['origin']['cm'].loc['pred 1','true 1'], 
#                         e['origin']['summary'].loc[0, 'value'], 
#                         e['origin']['summary'].loc[1, 'value'], 
#                         e['origin']['summary'].loc[2, 'value'], 
#                         e['origin']['summary'].loc[3, 'value'], 
#                         e['origin']['summary'].loc[4, 'value'], 
#                         e['origin']['summary'].loc[5, 'value']
#                        ]}

# for key in e.keys():
#     if key == 'origin':
#         continue
#     else:
#         n = int(key.split(', ')[0].split('=')[-1])
#         frac = int(float(key.split(', ')[1].split('=')[-1])*100)
#         experiment[key] = [n, frac, e[key]['cm'].loc['pred 0','true 0'], 
#                                     e[key]['cm'].loc['pred 0','true 1'], 
#                                     e[key]['cm'].loc['pred 1','true 0'], 
#                                     e[key]['cm'].loc['pred 1','true 1'], 
#                                     e[key]['summary'].loc[0, 'value'], 
#                                     e[key]['summary'].loc[1, 'value'], 
#                                     e[key]['summary'].loc[2, 'value'], 
#                                     e[key]['summary'].loc[3, 'value'], 
#                                     e[key]['summary'].loc[4, 'value'], 
#                                     e[key]['summary'].loc[5, 'value']]
        
# experiment = pd.DataFrame(experiment, index=['n', 'frac', 'pred 0|true 0','pred 0|true 1', 
#                                              'pred 1|true 0', 'pred 1|true 1', 'Sensitivity', 
#                                              'Recall', 'Specificity', 'Precision', 'Accuracy', 
#                                              'F1Score']).T


# In[143]:


# def plotExpSummary(col)
#     trace1 = {
#       'name': f'frac=75%', 
#       'type': 'scatter', 
#       'x': [5, 10, 20, 30, 40, 50], 
#       'y': experiment[experiment.frac==75][[col]].values.reshape((-1,)).tolist()
#     }
#     trace2 = {
#       'name': f'frac=50%', 
#       'type': 'scatter', 
#       'x': [5, 10, 20, 30, 40, 50], 
#       'y': experiment[experiment.frac==50][[col]].values.reshape((-1,)).tolist()
#     }
#     trace3 = {
#       'name': f'frac=25%', 
#       'type': 'scatter', 
#       'x': [5, 10, 20, 30, 40, 50], 
#       'y': experiment[experiment.frac==25][[col]].values.reshape((-1,)).tolist()
#     }

#     data = [trace1, trace2, trace3]
#     layout = {
#       'title': f'Ensemble Under Various Settings', 
#       'xaxis': {
#         'title': 'Number of Models per Algorithm', 
#         'zeroline': False
#       }, 
#       'yaxis': {
#         'title': 'Sample Rate', 
#         'zeroline': False,
#       },
#     }

#     fig = Figure(data=data, layout=layout)
#     py.iplot(fig, validate = False)

