
import pandas as pd
from sklearn.impute import SimpleImputer

house=pd.read_csv(r'train.csv',keep_default_na=False)
means = house.groupby('Neighborhood')['SalePrice'].mean()
def dataprocess(path,means):
    #housing dataset
    house=pd.read_csv(path,keep_default_na=False)
    print(house.shape)
    
    house.loc[:,'LotFrontage']=pd.to_numeric(house.loc[:,"LotFrontage"],errors='coerce' )
    house.loc[:,'TotalBsmtSF']=pd.to_numeric(house.loc[:,"TotalBsmtSF"],errors='coerce' )
    house.loc[:,'BsmtFinSF2']=pd.to_numeric(house.loc[:,"BsmtFinSF2"],errors='coerce' )
    house.loc[:,'BsmtFullBath']=pd.to_numeric(house.loc[:,"BsmtFullBath"],errors='coerce' )
    house.loc[:,'BsmtHalfBath']=pd.to_numeric(house.loc[:,"BsmtHalfBath"],errors='coerce' )
    house.loc[:,'BsmtFinSF1']=pd.to_numeric(house.loc[:,"BsmtFinSF1"],errors='coerce' )
    house.loc[:,'GarageCars']=pd.to_numeric(house.loc[:,"GarageCars"],errors='coerce' )
    house.loc[:,'GarageArea']=pd.to_numeric(house.loc[:,"GarageArea"],errors='coerce' )
    house.loc[:,'BsmtUnfSF']=pd.to_numeric(house.loc[:,"BsmtUnfSF"],errors='coerce' )
    LotFrontage=house['LotFrontage'].loc[:"LotFrontage"].values.reshape(-1,1)
    MasVnrType=house['MasVnrType'].loc[:"MasVnrType"].values.reshape(-1,1)
    Electrical=house['Electrical'].loc[:"Electrical"].values.reshape(-1,1)
    GarageYrBlt=house['GarageYrBlt'].loc[:"GarageYrBlt"].values.reshape(-1,1)
    MasVnrArea=house['MasVnrArea'].loc[:"MasVnrArea"].values.reshape(-1,1)
    Utilities=house['Utilities'].loc[:"Utilities"].values.reshape(-1,1)
    TotalBsmtSF=house['TotalBsmtSF'].loc[:"TotalBsmtSF"].values.reshape(-1,1)
    BsmtFinSF2=house['BsmtFinSF2'].loc[:"BsmtFinSF2"].values.reshape(-1,1)
    BsmtUnfSF=house['BsmtUnfSF'].loc[:"BsmtUnfSF"].values.reshape(-1,1)
    BsmtFullBath=house['BsmtFullBath'].loc[:"BsmtFullBath"].values.reshape(-1,1)
    BsmtHalfBath=house['BsmtHalfBath'].loc[:"BsmtHalfBath"].values.reshape(-1,1)
    BsmtFinSF1=house['BsmtFinSF1'].loc[:"BsmtFinSF1"].values.reshape(-1,1)
    GarageCars=house['GarageCars'].loc[:"GarageCars"].values.reshape(-1,1)
    GarageArea=house['GarageArea'].loc[:"GarageArea"].values.reshape(-1,1)
    
    imp_mean = SimpleImputer() 
    imp_mean = imp_mean.fit_transform(LotFrontage) 
    
    imp_fre1 = SimpleImputer(missing_values='NA',strategy="most_frequent") 
    imp_fre1 = imp_fre1.fit_transform(MasVnrType)
    
    imp_fre2 = SimpleImputer(missing_values='NA',strategy="most_frequent") 
    imp_fre2 = imp_fre2.fit_transform(Electrical)
    
    imp_fre3 = SimpleImputer(missing_values='NA',strategy="most_frequent") 
    imp_fre3 = imp_fre3.fit_transform(GarageYrBlt)
    
    imp_fre4 = SimpleImputer(missing_values='NA',strategy="most_frequent") 
    imp_fre4 = imp_fre4.fit_transform(MasVnrArea)
    
    imp_fre5 = SimpleImputer(missing_values='NA',strategy="most_frequent") 
    imp_fre5 = imp_fre5.fit_transform(Utilities)
    
    imp_fre6 = SimpleImputer() 
    imp_fre6 = imp_fre6.fit_transform(TotalBsmtSF)
    
    imp_fre7 = SimpleImputer() 
    imp_fre7 = imp_fre7.fit_transform(BsmtFinSF2)
    
    imp_fre8 = SimpleImputer(strategy="most_frequent") 
    imp_fre8 = imp_fre8.fit_transform(BsmtUnfSF)
    
    imp_fre9 = SimpleImputer() 
    imp_fre9 = imp_fre9.fit_transform(BsmtFullBath)
    
    imp_fre10 = SimpleImputer() 
    imp_fre10 = imp_fre10.fit_transform(BsmtHalfBath)
    
    imp_fre11 = SimpleImputer() 
    imp_fre11 = imp_fre11.fit_transform(BsmtFinSF1)
    
    imp_fre12 = SimpleImputer() 
    imp_fre12 = imp_fre12.fit_transform(GarageCars)
    
    imp_fre13 = SimpleImputer() 
    imp_fre13 = imp_fre13.fit_transform(GarageArea)
    
    
    house.loc[:,"LotFrontage"] = imp_mean
    house.loc[:,"MasVnrType"] = imp_fre1
    house.loc[:,"Electrical"] = imp_fre2
    house.loc[:,"GarageYrBlt"] = imp_fre3
    house.loc[:,"MasVnrArea"] = imp_fre4
    house.loc[:,"Utilities"] = imp_fre5
    house.loc[:,"TotalBsmtSF"] = imp_fre6
    house.loc[:,"BsmtFinSF2"] = imp_fre7
    house.loc[:,"BsmtUnfSF"] = imp_fre8
    house.loc[:,"BsmtFullBath"] = imp_fre9
    house.loc[:,"BsmtHalfBath"] = imp_fre10
    house.loc[:,"BsmtFinSF1"] = imp_fre11
    house.loc[:,"GarageCars"] = imp_fre12
    house.loc[:,"GarageArea"] = imp_fre13
    
    ###Int encoding
    LotShape={'Reg':0,'IR1':1,'IR2':2,'IR3':3}
    house['LotShape']=house['LotShape'].map(LotShape)
    
    Street={'Grvl':0,'Pave':1}
    house['Street']=house['Street'].map(Street)
    
    Utilities={'AllPub':1,'NoSeWa':0}
    house['Utilities']=house['Utilities'].map(Utilities)
    
    LandSlope={'Gtl':0,'Mod':1,'Sev':2}
    house['LandSlope']=house['LandSlope'].map(LandSlope)
    
    Condition1={'RRAn':0,'RRAe':0,'Artery':1,'Feedr':2,'RRNn':3,'RRNe':3,'Norm':4,'PosN':5,'PosA':6}
    house['Condition1']=house['Condition1'].map(Condition1)
    
    Condition2={'RRAn':0,'RRAe':0,'Artery':1,'Feedr':2,'RRNn':3,'RRNe':3,'Norm':4,'PosN':5,'PosA':6}
    house['Condition2']=house['Condition2'].map(Condition2)
    
    ExterQual={'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
    house['ExterQual']=house['ExterQual'].map(ExterQual)
    
    ExterCond={'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
    house['ExterCond']=house['ExterCond'].map(ExterCond)
    
    BsmtQual={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['BsmtQual']=house['BsmtQual'].map(BsmtQual)
    
    BsmtCond={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['BsmtCond']=house['BsmtCond'].map(BsmtCond)
    
    BsmtExposure={'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}
    house['BsmtExposure']=house['BsmtExposure'].map(BsmtExposure)
    
    BsmtFinType1={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
    house['BsmtFinType1']=house['BsmtFinType1'].map(BsmtFinType1)
    
    BsmtFinType2={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
    house['BsmtFinType2']=house['BsmtFinType2'].map(BsmtFinType2)
        
    HeatingQC={'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
    house['HeatingQC']=house['HeatingQC'].map(HeatingQC)
    
    CentralAir={'N':0,'Y':1}
    house['CentralAir']=house['CentralAir'].map(CentralAir)
    
    Electrical={'Mix':0,'FuseP':1,'FuseF':2,'FuseA':3,'SBrkr':4}
    house['Electrical']=house['Electrical'].map(Electrical)
    
    KitchenQual={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['KitchenQual']=house['KitchenQual'].map(KitchenQual)
    
    Functional={'NA':0,'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7}
    house['Functional']=house['Functional'].map(Functional)
    
    FireplaceQu={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['FireplaceQu']=house['FireplaceQu'].map(FireplaceQu)
    
    GarageFinish={'Fin':3,'RFn':2,'Unf':1,'NA':0}
    house['GarageFinish']=house['GarageFinish'].map(GarageFinish)
    
    GarageQual={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['GarageQual']=house['GarageQual'].map(GarageQual)
    
    GarageCond={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['GarageCond']=house['GarageCond'].map(GarageCond)
    
    PavedDrive={'N':0,'Y':1,'P':2}
    house['PavedDrive']=house['PavedDrive'].map(PavedDrive)
    
    PoolQC={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
    house['PoolQC']=house['PoolQC'].map(PoolQC)
    
    Fence={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0}
    house['Fence']=house['Fence'].map(Fence)
    
    
    ###one-hot encoding
    house = pd.get_dummies(house, columns=["Alley"], prefix="Alley")
    house = pd.get_dummies(house, columns=["MSZoning"], prefix="MSZoning")
    house = pd.get_dummies(house, columns=["LandContour"], prefix="LandContour")
    house = pd.get_dummies(house, columns=["LotConfig"], prefix="LotConfig")
    house = pd.get_dummies(house, columns=["BldgType"], prefix="BldgType")
    house = pd.get_dummies(house, columns=["HouseStyle"], prefix="HouseStyle")
    house = pd.get_dummies(house, columns=["RoofStyle"], prefix="RoofStyle")
    house = pd.get_dummies(house, columns=["Exterior1st"], prefix="Exterior1st")
    house = pd.get_dummies(house, columns=["Exterior2nd"], prefix="Exterior2nd")
    house = pd.get_dummies(house, columns=["MasVnrType"], prefix="MasVnrType")
    house = pd.get_dummies(house, columns=["RoofMatl"], prefix="RoofMatl")
    house = pd.get_dummies(house, columns=["GarageType"], prefix="GarageType")
    house = pd.get_dummies(house, columns=["Foundation"], prefix="Foundation")
    house = pd.get_dummies(house, columns=["Heating"], prefix="Heating")
    house = pd.get_dummies(house, columns=["MiscFeature"], prefix="MiscFeature")
    house = pd.get_dummies(house, columns=["SaleType"], prefix="SaleType")
    house = pd.get_dummies(house, columns=["SaleCondition"], prefix="SaleCondition")
    
    
    ###Target encoding
    #means = house.groupby('Neighborhood')['SalePrice'].mean()
    house['Neighborhood'] = house['Neighborhood'].map(means)
    house.to_csv(r'house.csv')
    return house

from sklearn.preprocessing import StandardScaler

def datascale(house):
    ss=StandardScaler()
    house_temp=house.values
    house_s=ss.fit_transform(house_temp)
    house_scaled=pd.DataFrame(house_s,columns=house.columns)
    #house_scaled.to_csv(r'house_scaled.csv')
    return house_scaled,ss

def comparetwodf(data1,data2):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # get all columns
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    
    # find extra colums
    extra_columns = columns_df1.symmetric_difference(columns_df2)
    
    # delete extra
    if extra_columns:
        for column in extra_columns:
            if column in df1.columns:
                df1 = df1.drop(columns=column)
            if column in df2.columns:
                df2 = df2.drop(columns=column)
    
    return df1,df2

###predition
pathtrain='train.csv'
pathtest='test.csv'
house_train=dataprocess(pathtrain,means)
house_test=dataprocess(pathtest,means)


house_train,house_test=comparetwodf(house_train, house_test)
house_scaled,ss=datascale(house_train)
house_test_scaled=ss.fit_transform(house_test)

X_train=house_scaled.loc[:,["LotFrontage",
"LotArea",
"Street",
"LotShape",
"Utilities",
"LandSlope",
"Neighborhood",
"Condition1",
"Condition2",
"OverallQual",
"OverallCond",
"YearBuilt",
"YearRemodAdd",
"MasVnrArea",
"ExterQual",
"ExterCond",
"BsmtQual",
"BsmtCond",
"BsmtExposure",
"BsmtFinType1",
"BsmtFinType2",
"HeatingQC",
"CentralAir",
"Electrical",
"BsmtFullBath",
"BsmtHalfBath",
"FullBath",
"HalfBath",
"BedroomAbvGr",
"KitchenAbvGr",
"KitchenQual",
"TotRmsAbvGrd",
"Functional",
"Fireplaces",
"FireplaceQu",
"GarageYrBlt",
"GarageFinish",
"GarageCars",
"GarageArea",
"GarageQual",
"GarageCond",
"PavedDrive",
"WoodDeckSF",
"OpenPorchSF",
"EnclosedPorch",
"3SsnPorch",
"ScreenPorch",
"PoolArea",
"PoolQC",
"Fence",
"MiscVal",
"MoSold",
"YrSold",
"HouseStyle_1.5Fin",
"HouseStyle_1.5Unf",
"HouseStyle_1Story",
"HouseStyle_2.5Unf",
"HouseStyle_2Story",
"HouseStyle_SFoyer",
"HouseStyle_SLvl",
"Exterior1st_AsbShng",
"Exterior1st_AsphShn",
"Exterior1st_BrkComm",
"Exterior1st_BrkFace",
"Exterior1st_CemntBd",
"Exterior1st_HdBoard",
"Exterior1st_MetalSd",
"Exterior1st_Plywood",
"Exterior1st_Stucco",
"Exterior1st_VinylSd",
"Exterior1st_Wd Sdng",
"Exterior1st_WdShing",
"Exterior2nd_AsbShng",
"Exterior2nd_AsphShn",
"Exterior2nd_Brk Cmn",
"Exterior2nd_BrkFace",
"Exterior2nd_CmentBd",
"Exterior2nd_HdBoard",
"Exterior2nd_ImStucc",
"Exterior2nd_MetalSd",
"Exterior2nd_Plywood",
"Exterior2nd_Stone",
"Exterior2nd_Stucco",
"Exterior2nd_VinylSd",
"Exterior2nd_Wd Sdng",
"Exterior2nd_Wd Shng",
"RoofMatl_CompShg",
"RoofMatl_Tar&Grv",
"RoofMatl_WdShake",
"RoofMatl_WdShngl",
"Heating_GasA",
"Heating_GasW",
"Heating_Grav",
"Heating_Wall",
"MiscFeature_Gar2",
"MiscFeature_NA",
"MiscFeature_Othr",
"MiscFeature_Shed",
"SaleCondition_Abnorml",
"SaleCondition_AdjLand",
"SaleCondition_Alloca",
"SaleCondition_Family",
"SaleCondition_Normal",
"SaleCondition_Partial"]]
#X_train1=house_scaled.loc[:,'Alley_Grvl':'SaleCondition_Partial']
#X_train=X_train0.join(X_train1)
ss1=StandardScaler()
y_train=ss1.fit_transform(house.loc[:,'SalePrice'].values.reshape(-1,1))

house_test_scaled=pd.DataFrame(house_test_scaled,columns=house_test.columns)
X_test=pd.DataFrame(house_test_scaled).loc[:,["LotFrontage",
"LotArea",
"Street",
"LotShape",
"Utilities",
"LandSlope",
"Neighborhood",
"Condition1",
"Condition2",
"OverallQual",
"OverallCond",
"YearBuilt",
"YearRemodAdd",
"MasVnrArea",
"ExterQual",
"ExterCond",
"BsmtQual",
"BsmtCond",
"BsmtExposure",
"BsmtFinType1",
"BsmtFinType2",
"HeatingQC",
"CentralAir",
"Electrical",
"BsmtFullBath",
"BsmtHalfBath",
"FullBath",
"HalfBath",
"BedroomAbvGr",
"KitchenAbvGr",
"KitchenQual",
"TotRmsAbvGrd",
"Functional",
"Fireplaces",
"FireplaceQu",
"GarageYrBlt",
"GarageFinish",
"GarageCars",
"GarageArea",
"GarageQual",
"GarageCond",
"PavedDrive",
"WoodDeckSF",
"OpenPorchSF",
"EnclosedPorch",
"3SsnPorch",
"ScreenPorch",
"PoolArea",
"PoolQC",
"Fence",
"MiscVal",
"MoSold",
"YrSold",
"HouseStyle_1.5Fin",
"HouseStyle_1.5Unf",
"HouseStyle_1Story",
"HouseStyle_2.5Unf",
"HouseStyle_2Story",
"HouseStyle_SFoyer",
"HouseStyle_SLvl",
"Exterior1st_AsbShng",
"Exterior1st_AsphShn",
"Exterior1st_BrkComm",
"Exterior1st_BrkFace",
"Exterior1st_CemntBd",
"Exterior1st_HdBoard",
"Exterior1st_MetalSd",
"Exterior1st_Plywood",
"Exterior1st_Stucco",
"Exterior1st_VinylSd",
"Exterior1st_Wd Sdng",
"Exterior1st_WdShing",
"Exterior2nd_AsbShng",
"Exterior2nd_AsphShn",
"Exterior2nd_Brk Cmn",
"Exterior2nd_BrkFace",
"Exterior2nd_CmentBd",
"Exterior2nd_HdBoard",
"Exterior2nd_ImStucc",
"Exterior2nd_MetalSd",
"Exterior2nd_Plywood",
"Exterior2nd_Stone",
"Exterior2nd_Stucco",
"Exterior2nd_VinylSd",
"Exterior2nd_Wd Sdng",
"Exterior2nd_Wd Shng",
"RoofMatl_CompShg",
"RoofMatl_Tar&Grv",
"RoofMatl_WdShake",
"RoofMatl_WdShngl",
"Heating_GasA",
"Heating_GasW",
"Heating_Grav",
"Heating_Wall",
"MiscFeature_Gar2",
"MiscFeature_NA",
"MiscFeature_Othr",
"MiscFeature_Shed",
"SaleCondition_Abnorml",
"SaleCondition_AdjLand",
"SaleCondition_Alloca",
"SaleCondition_Family",
"SaleCondition_Normal",
"SaleCondition_Partial"]]
#X_test1=pd.DataFrame(house_test_scaled).loc[:,'Alley_Grvl':'SaleCondition_Partial']
#X_test=X_test0.join(X_test1)
#y_test=house_test.loc[:,'SalePrice']


import numpy as np

theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
theta_best_df = pd.DataFrame(data=theta_best[np.newaxis, :][:, :, 0], columns=X_train.columns)

y_train_pred=X_train.dot(theta_best)
y_test_pred=ss1.inverse_transform(X_test.dot(theta_best))


###calculate MSE, R2

def f(X, theta):
    return X.dot(theta)

def mean_squared_error(theta, X, y):
    return 0.5*np.mean((y-f(X, theta))**2)
    
def RSquare(y,y_mean,y_pred):
    sst = np.sum((y - y_mean)**2)
    ssr = np.sum((y - y_pred)**2)
    r_squared = 1-(ssr/sst)
    return r_squared

print('R square=',RSquare(y_train,np.mean(y_train),y_train_pred))
print('MSE=',mean_squared_error(theta_best, X_train, y_train))


###visualize OHE
import matplotlib.pyplot as plt
fig=plt.figure()
house_train.loc[:,'Alley_Grvl':'SaleCondition_Partial'].hist()
beforeOHE=house.loc[:,["Alley",
"MSZoning",
"LandContour",
"LotConfig",
"BldgType",
"HouseStyle",
"RoofStyle",
"Exterior1st",
"Exterior2nd",
"MasVnrType",
"RoofMatl",
"GarageType",
"Foundation",
"Heating",
"MiscFeature",
"SaleType",
"SaleCondition"]]

for _ in beforeOHE:
    beforeOHE[_]=pd.factorize(beforeOHE[_])[0].astype(int)
beforeOHE.hist()



fig=plt.figure()
plt.rcParams['figure.figsize'] = [8, 4]
plt.xlabel('OverallQual')
plt.ylabel('Price')
plt.scatter(X_train.loc[:, ['OverallQual']], y_train)
#plt.scatter(X_test.loc[:, ['OverallQual']], y_test, color='red', marker='o')
plt.plot(X_train.loc[:, ['OverallQual']], y_train_pred, 'x', color='red', mew=3, markersize=8)
plt.legend(['Model', 'Prediction', 'Initial', 'New'])
plt.show()


###output prediction
output = pd.DataFrame(house_test['Id'],columns = ['Id'])
output = pd.concat([output,pd.DataFrame(y_test_pred,columns=['SalePrice'])],axis=1)
output.to_csv(r'output.csv',index=False)

'''
print(house)


imp_mode = SimpleImputer(strategy = "most_frequent")
data.loc[:,"Embarked"] = imp_mode.fit_transform(Embarked)
data.info()
'''
#imp = SimpleImputer(missing_values=np.nan, strategy='mean')