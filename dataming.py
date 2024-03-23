import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

def loader():   #读取数据
    filepath ="movies_dataset.csv" #使用的数据集文件地址
    df = pd.read_csv(filepath, header=0)
    return df

def loader1():  #读取缺失值数据
    filepath = "movies_dataset.csv"
    df = pd.read_csv(filepath, header=0, usecols = [1,4])
    print("------------------------------------------------------------------------")
    print("源数据：\n",df)
    return df

def count(str1,data1): #输出标称数据频数
    print(data1[str1].value_counts())

def fiveNumberandnull(str2,data2): #输出数值数据5数概括及缺失值的个数
    nums = data2[str2]
    nullnum = nums.isnull().sum()
    print("null:%d"%(nullnum))
    # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    nums = nums.dropna(axis = 0) #删除NaN值
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)
    print("Minimum:%d; Q1:%d; Median:%d; Q3:%d; Maximum:%d;"%(Minimum , Q1 , Median , Q3 , Maximum)) #都为整数

def PLOT1(str1,dt1): #绘制直方图
    hist = dt1[str1].hist(bins= 100) #显示在100个bins中
    plt.show()

def PLOT2(str2,dt2):    #绘制盒图
    dt2[str2].plot.box()
    plt.show()

def fun1(data1):    #将缺失部分剔除
    Data1 = data1.dropna(axis = 0)
    print("------------------------------------------------------------------------")
    print("将缺失部分剔除后的数据与相应的图：\n",data1)
    # 绘制直方图
    hist = Data1.hist(bins=100)  # 显示在100个bins中
    # 绘制盒图
    Data1.plot.box()
    plt.show()

def fun2(data2):    #用最高频率值来填补缺失值
    Data2 = data2.fillna(data2.mode()) #填充众数
    print("------------------------------------------------------------------------")
    print("用最高频率值来填补缺失值后的数据与相应的图：\n",data2)
    # 绘制直方图
    hist2 = Data2.hist(bins=100)  # 显示在100个bins中
    # 绘制盒图
    Data2.plot.box()
    plt.show()
    
def fun3(data3):    #通过属性的相关关系来填补缺失值
    #随机森林填充
    known = data3[data3['IMDb-rating'].notnull()]
    uknown = data3[data3['IMDb-rating'].isnull()]
    X = known.drop(columns=['IMDb-rating'])
    y = known['IMDb-rating']
    X = X.fillna(X.mean())

    rf = RandomForestRegressor()
    rf.fit(X, y)
    predicted = rf.predict(uknown.drop(columns=['IMDb-rating']))
    data3.loc[data3['IMDb-rating'].isnull(), 'IMDb-rating'] = predicted
    print("------------------------------------------------------------------------")
    print("通过属性的相关关系来填补缺失值后的数据与相应的图：\n",data3)
    # 绘制直方图
    hist = data3.hist(bins=100)  # 显示在100个bins中
    # 绘制盒图
    data3.plot.box()
    plt.show()

def fun4(data4):    #通过数据对象之间的相似性来填补缺失值
    #KNN算法填充
    knn_imputer = KNNImputer(n_neighbors=5)
    Data4 = pd.DataFrame(knn_imputer.fit_transform(data4))
    print("------------------------------------------------------------------------")
    print("通过数据对象之间的相似性来填补缺失值后的数据与相应的图：\n",data4)
    # 绘制直方图
    hist = Data4.hist(bins=100)  # 显示在100个bins中
    # 绘制盒图
    Data4.plot.box()
    plt.show()

if __name__ == "__main__":
    df = loader()
    #输出标称数据的每个可能的频数
    print("------------------------------------------------------------------------")
    print("industry频数：")
    count("industry",df)
    print("------------------------------------------------------------------------")
    print("appropriate_for频数：")
    count("appropriate_for",df)
    print("------------------------------------------------------------------------")
    print("director频数：")
    count("director",df)
    print("------------------------------------------------------------------------")
    print("language频数：")
    count("language",df)
    print("------------------------------------------------------------------------")
    print("posted_date频数：")
    count("posted_date",df)
    print("------------------------------------------------------------------------")
    print("release_date频数：")
    count("release_date",df)
    print("------------------------------------------------------------------------")
    print("id频数：")
    count("id",df)
    print("------------------------------------------------------------------------")

    #输出数值数据5数概括及缺失值的个数
    print("IMDb-rating数概括及缺失值的个数:")
    fiveNumberandnull("IMDb-rating",df)
    print("------------------------------------------------------------------------")
    print("downloads数概括及缺失值的个数:")
    fiveNumberandnull("downloads",df)

    #输出绘制的直方图和盒图
    print("------------------------------------------------------------------------")
    print("IMDb-rating与downloads的直方图：")
    PLOT1("IMDb-rating",df)
    PLOT1("downloads",df)
    print("------------------------------------------------------------------------")
    print("IMDb-rating与downloads的盒图：")
    PLOT2("IMDb-rating", df)
    PLOT2("downloads", df)

    #输出缺失值处理后的直方图与盒图
    df1 = loader1()  
    fun1(df1)
    fun2(df1)
    fun3(df1)
    fun4(df1)
