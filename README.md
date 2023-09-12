# House_Price_OLS
Pre-processing data with more than 80 attributes, using ordinary least squares (OLS), try to predict house prices on Kaggle dataset.  Implemented OLS from scratch without using external libraries or packages in the core regression process.

## Data Pre-processing

### (1) Dealing with missing values:

When using scikit-learn, it is assumed that all values in the array are numbers with meaning. There are two approaches for missing data: First, discard the entire row or column containing missing values. HoIver, this comes at the cost of losing potentially valuable data. Therefore, I choose the second method to infer missing values from a known data part, using sklearn.impute.SimpleImputer.

Read data description and find that NA has actual meaning in the following features and is not a missing value: Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature. In contrast, in features including LotFrontage, MasVnrType, and Electrical, NA represents missing values.

LotFrontage is continuous, so I use the mean value to replace missing values. MasVnrType and Electrical are categorical, so I use the most frequently occurring value to replace missing values.

### (2) Dealing with categorical values:

I need to use numbers to replace those data using letters. For ranked categorical variables, I use ordered numbers. For example, the Lotshape feature has four different kinds of shapes named IR1, IR2, IR3, Reg. I use 0 for Reg, 1 for IR1, 2 for IR2, and 3 for IR3 to make data in numerical form. In this case, from 0 to 3 means the irregularity of the shape gradually increases, as Reg means regular, IR1 means slightly irregular, IR2 means moderately irregular, and IR3 means irregular.

Pandas.map is used for ranked categorical variables encoding.

HoIver, some input data does not have any ranking for category values, which can lead to problems with predictions. So here, I use one-hot encoding to pre-process categorical values. For example, Alley, which indicates types of alley access to the property, has three categories: gravel, paved, and no alley access. I encode this feature into two features: IsGravel,  IsPaved, and NoAccess, when {0,0,1} means no alley access, {1,0,0} means gravel, and {0,1,0} means paved.

Pandas.get_dummies is used for one-hot encoding.

Considering that using one-hot encoding will increase dimensionality and lead to multicollinearity, I don't use this method to encode features with too many different values. For instance, I use Target Encoding to pre-process Neighborhood data, a feature meaning physical locations within Ames city limits.

### (3) Normalizing numerical values:
    I use scikit-learn to normalize data to have mean zero and standard deviation one. 


Here I calculate the R square is 0.856621, and MSE is 0.071689 for train dataset.
