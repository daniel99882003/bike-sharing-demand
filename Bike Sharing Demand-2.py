import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

# 1. 讀取資料
train = pd.read_csv('bike-sharing-demand/train.csv')

# train_X = train[['temp', 'atemp']].values
# train_y = train['count'].values

# 2. 處理時間類型資料
train['datetime'] = pd.to_datetime(train['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day_of_month'] = train['datetime'].dt.day
train['day_of_week'] = train['datetime'].dt.day_of_week
train['hour'] = train['datetime'].dt.hour

print(train.columns)

# # *target 分布分析
# train_WithoutOutliers = train[np.abs(train['count'] -
#                                      train['count'].mean()) <= (3 * train['count'].std())]
# print(train_WithoutOutliers.shape)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 4, 1)
# ax2 = fig.add_subplot(1, 4, 2)
# ax3 = fig.add_subplot(1, 4, 3)
# ax4 = fig.add_subplot(1, 4, 4)
# fig.set_size_inches(12, 5)
#
# sns.histplot(train_WithoutOutliers['count'], ax=ax1)
# sns.histplot(train['count'], ax=ax2)
# sns.histplot(np.log(train_WithoutOutliers['count']), ax=ax3)
# sns.histplot(np.log(train['count']), ax=ax4)
#
# ax1.set(xlabel='count', title='Distribution of count without outliers')
# ax2.set(xlabel='count', title='Distribution of count')
# ax3.set(xlabel='count', title='log of count without outliers')
# ax4.set(xlabel='count', title='log of count')
# plt.show()
#
# print('train_WithoutOutliers', train_WithoutOutliers['count'].skew(), train_WithoutOutliers['count'].kurt())
# print('train', train['count'].skew(), train['count'].kurt())
# print('train_WithoutOutliers_log', np.log(train_WithoutOutliers['count']).skew(),
#       np.log(train_WithoutOutliers['count']).kurt())
# print('train_log', np.log(train['count']).skew(), np.log(train['count']).kurt())

# 3. 觀察特徵值相關性
train_y = np.log(train['count'])
plt.figure(figsize=(16,12))
sns.heatmap(train.corr(), cmap='Purples', annot=True, linecolor='Green', linewidths=0.2)
plt.show()
print(train.corr()['count'].sort_values())




# 4. 切割訓練資料
drop_list = ['datetime', 'casual', 'registered', 'count']
train.drop(drop_list, axis=1, inplace=True)
print(train.columns)
x_train, x_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2, random_state=1)
print('x train :', x_train.shape, '\t\tx test :', x_test.shape)
print('y train :', y_train.shape, '\t\ty test :', y_test.shape)

# 5. 特徵值標準化
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 6. 使用模型
degree = 2
model_list = [('LinearRegression', LinearRegression()), ('KNN', KNeighborsRegressor(n_neighbors=4)),
              ('DecisionTree', DecisionTreeRegressor(random_state=1)),
              ('RandomForest', RandomForestRegressor(random_state=1, n_estimators=10))]

# ('poly', LinearRegression()),
# # ----------------------------------------
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
#
# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# r2s = metrics.r2_score(y_test, y_pred)
# me = metrics.max_error(y_test, y_pred)
#
# print('Mean Absolute Error:', mae)
# print('Mean Squared Error:', mse)
# print('R^ 2 Score:', r2s)
# print('Max Error:', me)
# # ----------------------------------------
# poly = PolynomialFeatures(degree=2)
# x_train_qua = poly.fit_transform(x_train)
# qua = LinearRegression()
# qua.fit(x_train_qua, y_train)
# y_pred = qua.predict(poly.fit_transform(x_test))
#
# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# r2s = metrics.r2_score(y_test, y_pred)
# me = metrics.max_error(y_test, y_pred)
#
# print('Mean Absolute Error:', mae)
# print('Mean Squared Error:', mse)
# print('R^ 2 Score:', r2s)
# print('Max Error:', me)

# # 評估效能
# def mt(mtlist, name, y_test, y_pred):
#     mae = metrics.mean_absolute_error(y_test, y_pred)
#     mse = metrics.mean_squared_error(y_test, y_pred)
#     r2s = metrics.r2_score(y_test, y_pred)
#     me = metrics.max_error(y_test, y_pred)
#     matric = [name, mae, mse, r2s, me]
#     mtlist.append(matric)
#

#
#
# mt_list = []
#
# poly = PolynomialFeatures(degree=2)
# x_train_qua = poly.fit_transform(x_train)
# qua = LinearRegression()
# qua.fit(x_train_qua, y_train)
# y_pred = qua.predict(poly.fit_transform(x_test))
# mt(mt_list, 'poly', y_test, y_pred)
#
# for name in model_list:
#     model = name[1]
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     mt(mt_list, name[0], y_test, y_pred)
#
# mt_df = pd.DataFrame(mt_list, columns=['name', 'mae', 'mse', 'r2s', 'me'])
# print(mt_df)
#
mt_list2 = []
for i in range(10, 40):
    model = RandomForestRegressor(random_state=1, n_estimators=i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mt(mt_list2, 'RandomForest' + str(i), y_test, y_pred)

mt_df2 = pd.DataFrame(mt_list2, columns=['name', 'mae', 'mse', 'r2s', 'me'])
print(mt_df2)

plt.plot(mt_df2['r2s'])
plt.show()
print(mt_df2['r2s'].max())
print(mt_df2['r2s'].argmax)

# 7. 準備test資料
test = pd.read_csv('bike-sharing-demand/test.csv')

test['datetime'] = pd.to_datetime(test['datetime'])
times = test['datetime']

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day_of_month'] = test['datetime'].dt.day
test['day_of_week'] = test['datetime'].dt.day_of_week
test['hour'] = test['datetime'].dt.hour

drop_list = ['datetime']
test.drop(drop_list, axis=1, inplace=True)
test.head(3)

sc = StandardScaler()
x_test = sc.fit_transform(test)

# 7. 確認最優模型後預測
model = RandomForestRegressor(random_state=1, n_estimators=31)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred[:5])
print([max(0, x) for x in np.exp(y_pred)][:5])

# 8. 格式化後輸出
Submission = pd.DataFrame({'datetime': times, 'count': [max(0, x) for x in np.exp(y_pred)]})
print(Submission.head())
Submission.to_csv('Submission1-2.csv')

Submission.set_index('datetime', inplace=True)
print(Submission.head())
Submission.to_csv('Submission.csv-2')
