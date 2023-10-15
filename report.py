import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

url = 'data/exams.csv'
df = pd.read_csv(url)

# 製作一個有效編碼器
one_hot = LabelBinarizer()

# 以一位有效編碼法對特徵編碼
gender_encoding = one_hot.fit_transform(df['gender'])
pre_encoding = one_hot.fit_transform(df['test preparation course'])
lunch_encoding = one_hot.fit_transform(df['lunch'])

# 將 one-hot encoding 結果存到新的DataFrame中
gender_df = pd.DataFrame(gender_encoding, columns=['gender'])
pre_df = pd.DataFrame(pre_encoding, columns=['prepare'])
lunch_df = pd.DataFrame(lunch_encoding, columns=['lunch'])

# 父母教育
high_school = ['some high school', 'high school']
tertiary_education = ["associate's degree", "bachelor's degree", 'college', 'some college']
postgraduate_education = ["master's degree"]

def categorize_education_level(level):
    if level in high_school:
        return 0
    elif level in tertiary_education:
        return 1 #大學教育程度
    elif level in postgraduate_education:
        return 2 #研究生教育程度
df['parental education'] = df['parental education'].apply(categorize_education_level)

#種族
def race_level(level):
    if level in "group A":
        return 1
    elif level in "group B":
        return 2
    elif level in "group C":
        return 3
    elif level in "group D":
        return 4
    elif level in "group E":
        return 5
    
df['race/ethnicity'] = df['race/ethnicity'].apply(race_level)

#數學成績標
lower_q=np.quantile(df['math score'],0.25,interpolation='lower')#下四分位数
higher_q=np.quantile(df['math score'],0.75,interpolation='higher')#上四分位数
def math_level(lev):
    if (lev <= lower_q):
        return 0 #後標
    elif (lower_q < lev and lev < higher_q):
        return 1 #均標
    elif (lev >= higher_q):
        return 2 #前標
    
math = df['math score'].apply(math_level)
math_df = pd.DataFrame({'math_level': math})

#讀成績標
lower_r=np.quantile(df['reading score'],0.25,interpolation='lower')#下四分位数
higher_r=np.quantile(df['reading score'],0.75,interpolation='higher')#上四分位数
def read_level(lev):
    if (lev <= lower_r):
        return 0 #後標
    elif (lower_r < lev and lev < higher_r):
        return 1 #均標
    elif (lev >= higher_r):
        return 2 #前標
    
df['reading score'] = df['reading score'].apply(read_level)

#寫成績標
lower_w=np.quantile(df['writing score'],0.25,interpolation='lower')#下四分位数
higher_w=np.quantile(df['writing score'],0.75,interpolation='higher')#上四分位数
def writing_level(lev):
    if (lev <= lower_w):
        return "0" #後標
    elif (lower_w < lev and lev < higher_w):
        return "1" #均標
    elif (lev >= higher_w):
        return "2" #前標
    
df['writing score'] = df['writing score'].apply(writing_level)

# 合併成一個DataFrame
df.drop(['gender', 'test preparation course', 'lunch'], axis=1, inplace=True)
df = pd.concat([gender_df, pre_df, lunch_df, df, math_df], axis=1)

#按照parental education分组
features = df[['gender', 'prepare', 'lunch', 'race/ethnicity', 'reading score', 'writing score']]
target = df['math_level']

results = {
    'edu_level': [],
    'low_achievements': [],
    'high_achievements': []
}

for level, group in df.groupby('parental education'):
    group_features = features.loc[group.index][group['parental education'] == level]
    group_target = target[group.index][group['parental education'] == level]    

    #分訓練集和測試集
    group_features_train, group_features_test, \
    group_target_train, group_target_test = \
    train_test_split(group_features, group_target, \
    test_size=0.3, random_state=42)

    # 產生隨機林迴歸器物件，訓練模型
    randomforest = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        max_depth=10,
        random_state=42
    )

    randomforest.fit(group_features_train, group_target_train)

    # 使用訓練資料預測
    pred_df = randomforest.predict(group_features_test)

    # 預測成功的比例
    if(level == 0 ):
        edu_level = '高中教育程度'
        edu_level_title = 'high school'
    elif(level == 1):
        edu_level = '大學教育程度'
        edu_level_title = 'tertiary'
    else:
        edu_level = '研究所程度'
        edu_level_title = 'postgraduate'

    feature_importances = np.round(randomforest.feature_importances_, 2)

    print('訓練集: ',randomforest.score(group_features_train, group_target_train))
    print('測試集: ',randomforest.score(group_features_test, group_target_test))
    print('特徵: [gender, prepare, lunch, race/ethnicity, reading score, writing score]')
    print('特徵重要程度: ',feature_importances)
    
    # 印出結果
    class_names = ['high', 'normal', 'low']
    achievement_low = np.mean(pred_df == 0)
    achievement_high = np.mean(pred_df == 2)
    
    results['edu_level'].append(edu_level_title)
    results['low_achievements'].append(achievement_low)
    results['high_achievements'].append(achievement_high)
    
    print(f"父母或監護人教育背景：{edu_level}")
    print("學生的下一個數學成績預測結果：前標 ->",round(achievement_high, 2),"/ 後標 ->", round(achievement_low,2))

    # 產生邏輯迴歸
    classifier = LogisticRegression()

    # 訓練模型並作預測
    model = classifier.fit(group_features_train, group_target_train)
    target_predicted = model.predict(group_features_test)

    # 產生分類報告
    print("------------------------ 分類報告 ------------------------")
    print(classification_report(group_target_test,
                                target_predicted,
                                target_names=class_names))
    print("---------------------------------------------------------")
    print()

colors = ['orange', 'green', 'blue']
labels = ['High School', 'Tertiary', 'Postgraduate']

# 直條圖
plt.bar(results['edu_level'], results['low_achievements'], color=colors)
plt.title('Math Score: Low Achievements \nby Parental Education Level')
plt.xlabel('Parental Education Level')
plt.ylabel('Low Achievement Proportion')
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
plt.legend(handles=legend_patches)
plt.show()
   
# 直條圖
plt.bar(results['edu_level'], results['high_achievements'], color=colors)
plt.title('Math Score: High Achievements \nby Parental Education Level')
plt.xlabel('Parental Education Level')
plt.ylabel('High Achievement Proportion')
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
plt.legend(handles=legend_patches)
plt.show()

#Silhouette Score
features, _ = make_blobs(n_samples=1000,
                         n_features=10,
                         centers=2,
                         cluster_std=0.5,
                         shuffle=True,
                         random_state=1)

model = KMeans(n_clusters=2, random_state=1).fit(features)

target_predicted = model.labels_

score = silhouette_score(features, target_predicted)

print("Silhouette Score:", score)




