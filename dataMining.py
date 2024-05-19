# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:14:36 2023

@author: zihong
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.tree import export_graphviz
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import mlxtend
from mlxtend.plotting import plot_decision_regions
import pydotplus  
import warnings
warnings.simplefilter('ignore')

def transData():
    df = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/crop1.csv')
    df.head()
    # print(df)
    transformed_data = {}
    # print(df.iterrows())
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # print(row)
        area = row["Area"]
        item = row["Item"]
        element = row["Element"]
        year = row["Year"]
        unit = row["Unit"]
        value = row["Value"]
        
        #transformed data  key值
        key = (area, year)
        # 檢查有沒有同樣key值存在
        if key in transformed_data:
            # 依照老師/助教建議, 考慮後依照題目本身，只呈現農作物組合有關特徵 拿掉農作物對應農地面積的特徵
            # if element == "Area harvested":
            #     transformed_data[key][item+"_ha"] = value
            #     # 依照老師/助教建議將作物組合之外特徵值拿掉
            #     # if value>0:
            #         # transformed_data[key]["total_ha"] += value
            if element == "Production":
                transformed_data[key][item] = value   
                # 依照老師/助教建議將作物組合之外特徵值拿掉
                # if value>0:
                    # transformed_data[key]["itm_count"] += 1
                 
        else:
            # 依照老師/助教建議, 考慮後依照題目本身，只呈現農作物組合有關特徵 拿掉農作物對應農地面積的特徵
            # if element == "Area harvested":
            #     transformed_data[key] = {item+"_ha": value}
            #     # 因不確定資料 Area harvested/Production 排序
            #     # 依照老師/助教建議將作物組合之外特徵值拿掉
            #     # transformed_data[key]["total_ha"] = 0
            #     # transformed_data[key]["itm_count"] = 0  
            if element == "Production":
                transformed_data[key] = {item: value}
                # transformed_data[key]["total_ha"] = 0
                # transformed_data[key]["itm_count"] = 0 
          
    # print(transformed_data)
    # Convert the dictionary to a DataFrame
    df_transformed = np.array([], dtype=object)
    df_transformed = pd.DataFrame.from_dict(transformed_data, orient="index").reset_index()
    
    # # Rename the columns
    df_transformed.columns = ["Area", "Year"] + df_transformed.columns[2:].tolist()
    # 檢查每個欄位是否都是NaN，並刪除所有都是NaN的欄位
    df_transformed = df_transformed.loc[:, ~df_transformed.isna().all()]
    
    # 檢查每個欄位是否包含NaN值，並將NaN值替換為0
    df_transformed = df_transformed.fillna(0)
    # 配合world bank資料 , 'Area' 重命名為 'Region'
    df_transformed = df_transformed.rename(columns={'Area': 'Region'})
    # 另存為csv檔
    df_transformed.to_csv("newCrop_new.csv", index=False)

# 決策樹
def doDecisionTreeClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 宣告決策樹
    clf = DecisionTreeClassifier(max_depth=6)
    # 訓練
    clf.fit(X_train, y_train)
    # 預測
    y_pred = clf.predict(X_test)
    # 畫圖
    dot_data = export_graphviz(clf, out_file=None,
                    feature_names = crop.columns[:-1],
                    class_names = y.unique().astype(str),
                    rounded = True, proportion = False,
                    precision = 2, filled = True)
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_pdf("DTtree.pdf") 
    # 正確率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
# 隨機森林
def doRandomForestClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 宣告隨機森林
    rf = RandomForestClassifier(n_estimators=2, max_depth=6, random_state=1)
    # 訓練
    rf.fit(X_train, y_train)
    # 把每一個變數特徵的重要性列出，從大排到小
    ipt = rf.feature_importances_
    ipt_sort = np.argsort(ipt)[::-1]
    print('隨機森林 重要度排序')
    for i, f in enumerate(range(X_train.shape[1])):
        print(f'{i+1:>2d}) {crop.columns[ipt_sort[f]]:<30s} {ipt[ipt_sort[i]]:.4f}')
        
    result=''
    # 畫圖
    for i, per_rf in enumerate(rf.estimators_):
        dot_data = export_graphviz(per_rf, out_file=None, 
                             feature_names=crop.columns[:-1],  
                             class_names=y.unique().astype(str),
                             filled=True, rounded=True,  
                             special_characters=True)  
        graph = pydotplus.graph_from_dot_data(dot_data)  
        graph.write_pdf("RF_"+str(i+1)+"DTtree.pdf") 
        # 預測
        y_pred = per_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result += "RF_"+str(i+1)+"DTtree score==>"+str(accuracy)+"   "
    return result
# LinearRegression
def doTrainByLr(X, y):
    # 一般來說，建議 8/2 或者 7/3
    # 希望每次執行的結果一致，可以給予 random_state 一個固定的數字；例如，random_state = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    # 權重
    lr.coef_
    # 截距
    lr.intercept_
    # 進行測試
    lr.predict(X_test)
    # 一般可以接受的門檻值大約是 0.5~0.6
    lr.score(X_test, y_test)
    scores = cross_val_score(lr, X, y, cv=3, scoring='r2')
    result = scores.mean()
    return result

# Polynomial
def doTrainByPf(X, y):
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)
    lr = LinearRegression(normalize=True)
    lr.fit(Xp, y)
    lr.score(Xp, y)
    sm = SelectFromModel(lr, threshold=10)
    Xt = sm.fit_transform(Xp, y)
    result = sm.estimator_.score(Xp, y)
    return result  

def doTrainNN(X, y):
    # 一般來說，建議 8/2 或者 7/3
    # 希望每次執行的結果一致，可以給予 random_state 一個固定的數字；例如，random_state = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = Sequential()
    # model.add(Dense(500, activation='tanh', input_shape=(X.shape[1], )))
    # model.add(Dense(500, input_shape=(3, )))
    model.add(Dense(700, activation='softmax', input_shape=(X.shape[1], )))
    # model.add(Dense(500, activation='softmax', input_shape=(6, )))
    model.add(Dense(1, activation='tanh'))
    # model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=30)
    history = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    print('=== end NN model ===')
    # print(history)
    result = history
    # 計算準確率
    accuracy = result[1] * 100
    print(f'準確率: {accuracy:.2f}%')
    
    # 將模型預測的機率轉換為二元分類預測值
    predictions_binary = np.round(predictions)
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_test, predictions_binary)
    
    # 繪製熱力圖
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    
    # 設定標題和軸標籤
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 顯示圖形
    plt.show()
    return result
# SVM 
def onlySVM(c_num,X, Y, kernel_do='rbf'):
    print('begin onlySVM')
    # 課本範例測試用
    # nb_samples = 500
    # X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    svc = SVC(C=c_num, kernel=kernel_do)
    cross_score = cross_val_score(svc, X, Y, scoring='accuracy', cv=10).mean()
    print('cross_score==>'+ str(cross_score))
    svc.fit(X_train, Y_train)
    svc_score = svc.score(X_test, Y_test)
    print('svc_score===>'+ str(svc_score))
    v_ahspe = svc.support_vectors_.shape
    print('vectors.shape===>'+ str(v_ahspe))
    print('end onlySVM')
# PCA
def doPCA(X,Y):
    # PCA降維
    print("PCA前",X.shape)
    pca = PCA(n_components=2)  # 保持90%訊息
    X_pca = pca.fit_transform(X,Y)

    # 投影後的特徵維度的方差比例
    print("投影後的特徵維度的方差比例",pca.explained_variance_ratio_)
    # 投影後的特徵維度的方差
    print("投影後的特徵維度的方差",pca.explained_variance_)
    print("PCA後結果",X_pca.shape)
    print("PCA後結果",X_pca[:3])

    # 獲取主成分的權重矩陣
    component_weights = pca.components_
    print("主成分的權重矩陣",component_weights)

    # 取得原始特徵名稱
    feature_names = crop.columns
    # 降維後data
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # 降維後 散布圖
    plt.scatter(df_pca['PC1'], df_pca['PC2'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Dimensionality Reduction')
    plt.show()
    return df_pca[['PC1', 'PC2']]
# Linear Discriminant Analysis (LDA)
def doLDA(X,Y):
    # LDA
    print("=== do LDA===")
    lda = LDA(n_components=2)
    # Ensure y_numpy is of the correct data type
    y_numpy = np.array(Y.values, dtype=np.int_)
    X_train_lda = lda.fit_transform(X,y_numpy)

    # 用一個LogisticRegression 演算法來評分
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_numpy)
    plot_decision_regions(X_train_lda, y_numpy, clf=lr, scatter_kwargs={})
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
def plotBoxPic(column_name):
    crop.boxplot(column=column_name)
    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.title('Boxplot of ' + column_name)
    plt.show()
    # 繪製指定欄位的盒形圖
# 主程式
# 將資料整理成 以地區 年分為key 各種農作物為其特徵  並產出newCrop_new.csv 方便PPT截圖呈現
# transData()

# 匯入資料
df = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/crop1.csv')
crop = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/newCrop_new.csv')
crop.head()
gap_data = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/property.csv')
# Poverty Line 都取 world bank 設定$2.15美元 日常維生困難之族群 
gap_data = gap_data.drop("Poverty Line", axis = 1)
rate_data = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/rate.csv')
rate_data = rate_data.drop("Poverty Line", axis = 1)
watts_data = pd.read_csv('D:/KC_AOI_Project/wrk/study_wrk/watts.csv')
watts_data = watts_data.drop("Poverty Line", axis = 1)

# 畫圖出來看資料長怎樣 找靈感
unique_units = df['Unit'].unique()
print("不同的值數量：", unique_units)
# df.isnull().sum()
# 看一下已開發國家 主要農作物:小麥 大麥 油菜籽
# 耕作面積雖起起伏伏但總體往下
df_ger = df[df['Area']=='Germany']
df_harvest = df_ger[(df_ger['Year']==2020)&(df_ger['Element']=='Area harvested')][['Item','Value']].sort_values(by='Value' ,ascending=False)
df_harvest = df_harvest.dropna(how='any')
df_harvest = df_harvest[df_harvest['Value']!=0]
plt.pie(df_harvest['Value'], radius=1.5, labels=df_harvest['Item'])
plt.show()
# 直條圖  圓餅圖太雜  看個前10多的
sns.set_theme(style = 'white',rc={'axes.facecolor':'white', 'figure.facecolor':'#F5C26F'})
plt.figure(figsize = (10,6))
top_15 = df_harvest.iloc[:10]
sns.barplot(data = top_15, y='Item', x = 'Value', palette = 'icefire')
plt.title('Top 10 crops with the highest yield in Germany 2020')
plt.xticks(rotation = 45)
plt.show()

# 折線圖  表示耕作面積變化
df_tot_area = df_ger[(df_ger['Element']=='Area harvested')][['Item','Year','Value']]
df_tot_area = df_tot_area.groupby('Year',as_index=False)['Value'].sum()

fig , ax = plt.subplots(figsize=(15,8))
ax       = sns.lineplot(data=df_tot_area, x='Year', y='Value', marker='o', linewidth = 2, color='#d52941')
sns.despine(offset=10)
sns.set_style({'axes.grid': True, 'axes.facecolor':'#faedcd', 'figure.facecolor':'#faedcd',
               "ytick.color":'#0f4920', "xtick.color":'#0f4920',  
               "grid.linestyle": ":", 'grid.color':'#0f4920',
               'font.family':'monospace','font.monospace' :'Times New Roman'})

plt.title("Yearly Harvested Area Change", size=16, y=1.07)
plt.ylabel("Yield ", size=12 ,labelpad=20)
plt.xlabel("Year", size=12, labelpad=20)
plt.show()
# 看一下開發中國家 
# 2020y資料 主要農作物為 玉米澱粉 豆子 茶葉
# 耕作面積逐年明顯提升
df_ger = df[df['Area']=='Kenya']
df_harvest = df_ger[(df_ger['Year']==2020)&(df_ger['Element']=='Area harvested')][['Item','Value']].sort_values(by='Value' ,ascending=False)
df_harvest = df_harvest.dropna(how='any')
df_harvest = df_harvest[df_harvest['Value']!=0]
plt.pie(df_harvest['Value'], radius=1.5, labels=df_harvest['Item'])
plt.show()

# 直條圖  圓餅圖太雜  看個前10多的
sns.set_theme(style = 'white',rc={'axes.facecolor':'white', 'figure.facecolor':'#F5C26F'})
plt.figure(figsize = (10,6))
top_15 = df_harvest.iloc[:10]
sns.barplot(data = top_15, y='Item', x = 'Value', palette = 'icefire')
plt.title('Top 10 crops with the highest yield in Kenya 2020')
plt.xticks(rotation = 45)
plt.show()

# 折線圖  表示耕作面積變化
df_tot_area = df_ger[(df_ger['Element']=='Area harvested')][['Item','Year','Value']]
df_tot_area = df_tot_area.groupby('Year',as_index=False)['Value'].sum()

fig , ax = plt.subplots(figsize=(15,8))
ax       = sns.lineplot(data=df_tot_area, x='Year', y='Value', marker='o', linewidth = 2, color='#d52941')
sns.despine(offset=10)
sns.set_style({'axes.grid': True, 'axes.facecolor':'#faedcd', 'figure.facecolor':'#faedcd',
               "ytick.color":'#0f4920', "xtick.color":'#0f4920',  
               "grid.linestyle": ":", 'grid.color':'#0f4920',
               'font.family':'monospace','font.monospace' :'Times New Roman'})

plt.title("Yearly Harvested Area Change", size=16, y=1.07)
plt.ylabel("Yield ", size=12 ,labelpad=20)
plt.xlabel("Year", size=12, labelpad=20)
plt.show()

# 全世界貧富差距越來愈大 看一下全世界收穫面積有沒有增加
# 耕作面積越來越多
df_ger = df
# 折線圖  表示耕作面積變化
df_tot_area = df_ger[(df_ger['Element']=='Area harvested')][['Item','Year','Value']]
df_tot_area = df_tot_area.groupby('Year',as_index=False)['Value'].sum()

fig , ax = plt.subplots(figsize=(15,8))
ax       = sns.lineplot(data=df_tot_area, x='Year', y='Value', marker='o', linewidth = 2, color='#d52941')
sns.despine(offset=10)
sns.set_style({'axes.grid': True, 'axes.facecolor':'#faedcd', 'figure.facecolor':'#faedcd',
               "ytick.color":'#0f4920', "xtick.color":'#0f4920',  
               "grid.linestyle": ":", 'grid.color':'#0f4920',
               'font.family':'monospace','font.monospace' :'Times New Roman'})

plt.title("World Yearly Harvested Area Change", size=16, y=1.07)
plt.ylabel("Yield ", size=12 ,labelpad=20)
plt.xlabel("Year", size=12, labelpad=20)
plt.show()

# 資料前處理
# 合併資料框
# gap_data 該年地區內貧窮落差
# 依照老師/助教建議, 考慮後依照題目本身，只呈現農作物組合有關特徵 拿掉Poverty Gap, Watts_Index的特徵
# crop = pd.merge(crop, gap_data, on=['Year', 'Region'], how='outer')
# rate_data 該年與全球其他地區對比後取得的貧窮程度
crop = pd.merge(crop, rate_data, on=['Year', 'Region'], how='outer')
# watts_data 該年與全球其他地區對比後取得的沙瓦特指數 , 用以表示氣候穩定程度 ,越接近0代表氣候變化大
# crop = pd.concat([crop,watts_data],axis=0)
# 以'Year','Region'為index去除重複資料
crop.drop_duplicates(subset = ['Year','Region'])
print("整理後的資料crop：", crop.shape)

# 根據 Region 分組，並將每個分組中的 NaN 值填補為該分組的中位數
# 依照老師/助教建議, 考慮後依照題目本身，只呈現農作物組合有關特徵 拿掉Poverty Gap, Watts_Index的特徵
# crop['Poverty Gap'].fillna(value=crop.groupby('Region')['Poverty Gap'].transform('median'), inplace=True)
crop['Poverty rate'].fillna(value=crop.groupby('Region')['Poverty rate'].transform('median'), inplace=True)
# 因為error msg顯示欄位值內有非數字值 無法取中位數,平均數 所以直接先補0
# crop['Watts Index'] = crop['Watts Index'].fillna(0)
# 剩下欄位為農作物數量 所以缺失值視為沒產出補0
crop.fillna(0, inplace=True)
# 檢查table中有沒有任何欄位有NaN值
nan_check = crop.isnull().any()
print("檢查table中有沒有任何欄位有NaN值：",nan_check)

# 定義區間範圍和對應的類別標籤  因Poverty rate是該年與全球其他地區對比後取得的貧窮程度,以此來label >=0且<1為1等,>=1且<2為2等以此類推
bins = [0,1,2,3,4,5,6,7,8,9,10,float('inf')]
labels =[0,1,2,3,4,5,6,7,8,9,10]
# 以Poverty rate 使用 pd.cut() 創建 Y 欄位
crop['Y'] = pd.cut(crop['Poverty rate'], bins=bins, labels=labels, right=False)
# 移除 Poverty rate 列
crop = crop.drop('Poverty rate', axis=1)
# crop = crop.rename(columns={'Watts Index': "Watts_Index"})
# print(crop)


# 删除Region值為0的行
crop = crop.loc[crop['Region'] != 0]
# 最終整理完的crop 另存為csv檔
# 產完先mark 要用再開
# crop.to_csv("newCrop_final.csv", index=False)
# 建立一個空的字典來儲存結果
region_counts = {}

# 使用for迴圈計算Y值從0到10的不重複的Region數量
for y in range(11):  # 從0到10
# 因為資料中每個國家有不同年度的資料 , 所以指定其中一年來看國家數量及國家T0-T10分布的數量
    subset = crop[(crop['Y'] == y) & (crop['Year'] == 2020)]
    unique_regions = subset['Region'].nunique()
    region_counts[y] = unique_regions

# 使用for迴圈印出結果
totoal_count = 0
for y, count in region_counts.items():
    totoal_count += count
    print("Y={}: 不重複的Region數量: {}".format(y, count))
print(totoal_count)

# 特徵選擇
X = crop.drop('Y', axis=1)
y = crop['Y']  # 假設"Y"是目標變數
# 印出X資料,檢查有無異常
# 發現合併資料時有產生 Region值為0的資料
# 欄位中 -取代成空格
mask = X.eq('-').any()
X['Region'] = X['Region'].replace('-', ' ')
# 依照老師/助教建議, 考慮後依照題目本身，只呈現農作物組合有關特徵 拿掉Poverty Gap, Watts_Index的特徵
# X['Watts_Index'] = X['Watts_Index'].replace('-', 0)
print(mask)
# 將'Region'欄位進行標籤編碼
encoding_dict = {region: i for i, region in enumerate(X['Region'])}
X['Region'] = [encoding_dict[region] for region in X['Region']]


# 繪製指定欄位的盒形圖
cols_with_missing_values=["Ramie","Papayas","Vanilla","Wheat","Apples"]
for col in cols_with_missing_values:
    plt.figure(figsize=(10,5))
    sns.boxplot(X[col])
    plt.title(col)
    plt.show()

# 關聯度分析
# correlation 計算
corr = crop.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r")
# 進行資料分割
# 正規化
# 儲存的參數較少，不需要考慮減少模型大小與記憶體需求 所以normalize選擇用l2 避免參數被縮減得太快而且約束至0 
no = Normalizer()
X_scaled = no.fit_transform(X)
df = pd.DataFrame(X_scaled)
df_cor = df.corr()

# 利用VarianceThreshold 移除低變異數的特徵 方差定義為至少大於 10-6 
# 僅檢視 因為想保留所有特徵值進PCD 及隨機森林
vt = VarianceThreshold(threshold=1e-06)
X_t = vt.fit_transform(X_scaled)
X_t.var(axis=0)
removed_features = np.logical_not(vt.get_support())

# Get the indices of the removed features
removed_indices = np.where(removed_features)[0]
# Get the names of the removed features
removed_feature_names = [X.columns[i] for i in removed_indices]
print('移除低變異數的特徵')
print("Removed Features:", removed_feature_names)




# 6. 訓練集和測試集劃分
X = X_scaled
X_pca = doPCA(X, y)
doLDA(X, y)

# 用原本的資料特徵
# DecisionTree
dtc_score = doDecisionTreeClassifier(X, y)
rf_score = doRandomForestClassifier(X, y)
# Polynomial
pf_score = doTrainByPf(X, y)
# NN model
NN_score = doTrainNN(X, y)
# # LinearRegression
lr_score = doTrainByLr(X, y)
print("DecisionTree Score==>:", dtc_score)
print(rf_score)
print('Polynomial Score==>'+ str(pf_score))
print('NNScore==>'+ str(NN_score))
print('LinearRegression Score==>'+ str(lr_score))
# SVM
onlySVM(8, X, y)
# 用PCA後得到的兩個資料特徵PC1 PC2
# 得到結果好像較差
X = X_pca
# Polynomial
pf_score = doTrainByPf(X, y)
# NN model
NN_score = doTrainNN(X, y)
# # LinearRegression
lr_score = doTrainByLr(X, y)
print('after PCA Polynomial Score==>'+ str(pf_score))
print('after PCA NNScore==>'+ str(NN_score))
print('after PCA LinearRegression Score==>'+ str(lr_score))
# SVM
print('after PCA onlySVM==>')
onlySVM(8, X, y)


