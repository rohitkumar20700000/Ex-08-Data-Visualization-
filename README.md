# Ex-08-Data-Visualization

## AIM
To Perform Data Visualization on the given dataset and save the data to a file.  

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4
Apply data visualization techniques to identify the patterns of the data.


## Data Pre-Processing

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```
df=pd.read_csv("Superstore.csv",encoding='unicode_escape')
df
```
```
df.head()
```
```
df.info()
```
```
df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df
```
```
df.isnull().sum()
```
```
#detecting and removing outliers in current numeric data
plt.figure(figsize=(8,8))
plt.title("Data with outliers")
df.boxplot()
plt.show()
```
```
plt.figure(figsize=(8,8))
cols = ['Sales','Quantity','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```

## Which Segment has Highest sales?
```
sns.lineplot(x="Segment",y="Sales",data=df,marker='o')
plt.title("Segment vs Sales")
plt.xticks(rotation = 90)
plt.show()
```
```
sns.barplot(x="Segment",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()
```
## Which City has Highest profit?
```
df.shape
df1 = df[(df.Profit >= 60)]
df1.shape
```
```
plt.figure(figsize=(30,8))
states=df1.loc[:,["City","Profit"]]
states=states.groupby(by=["City"]).sum().sort_values(by="Profit")
sns.barplot(x=states.index,y="Profit",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("City")
plt.ylabel=("Profit")
plt.show()
```

## Which ship mode is profitable?
```
sns.barplot(x="Ship Mode",y="Profit",data=df)
plt.show()
```
```
sns.lineplot(x="Ship Mode",y="Profit",data=df)
plt.show()
```
```
sns.violinplot(x="Profit",y="Ship Mode",data=df)
```
```
sns.pointplot(x=df["Profit"],y=df["Ship Mode"])
```
## Sales of the product based on region.
```
states=df.loc[:,["Region","Sales"]]
states=states.groupby(by=["Region"]).sum().sort_values(by="Sales")
sns.barplot(x=states.index,y="Sales",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("Region")
plt.ylabel=("Sales")
plt.show()
```
```
df.groupby(['Region']).sum().plot(kind='pie', y='Sales',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)
```
## Find the relation between sales and profit.
```
df["Sales"].corr(df["Profit"])
```
```
df_corr = df.copy()
df_corr = df_corr[["Sales","Profit"]]
df_corr.corr()
```
```
sns.pairplot(df_corr, kind="scatter")
plt.show()
```
## Heatmap
```
df4=df.copy()

#encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])

#scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])

#Heatmap
plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()
```
## Find the relation between sales and profit based on the following category.

### Segment
```
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment"]]
df_corr.corr()
```
### City
```
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","City"]]
df_corr.corr()
```
### States
```
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","State"]]
df_corr.corr()
```
### Segment and Ship Mode
```
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment","Ship Mode"]]
df_corr.corr()
```
### Segment, Ship mode and Region
```
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment","Ship Mode","Region"]]
df_corr.corr()
```

# OUPUT
## Data Pre-Processing
![df](https://user-images.githubusercontent.com/119160414/236831725-fbd5464b-31fb-49fa-9dbe-433ffb20fa0e.png)
![head](https://user-images.githubusercontent.com/119160414/236832106-359c1d65-8944-47d9-8946-cd932c81e204.png)
![info](https://user-images.githubusercontent.com/119160414/236832133-7ea0442a-71c5-4f08-8683-dc626aaae506.png)

![null](https://user-images.githubusercontent.com/119160414/236832176-2104e9f6-6335-402a-9df8-9534300663f4.png)

![shape](https://user-images.githubusercontent.com/119160414/236832223-11266af2-c398-401e-b741-a44cd149aa22.png)
![updateddf](https://user-images.githubusercontent.com/119160414/236832270-43d05f71-6f75-4187-a71b-6fd632792bb2.png)
![withoutliers](https://user-images.githubusercontent.com/119160414/236832321-77c66a52-cc90-4977-b9c1-3272e897b5c8.png)
![withoutoutliers](https://user-images.githubusercontent.com/119160414/236832366-7485124d-7fb9-4b52-9365-1b9b2d68ce7b.png)
## Which Segment has Highest sales?
![1](https://user-images.githubusercontent.com/119160414/236832404-8caef733-799b-4bea-8741-f0123b154f6d.png)

![2](https://user-images.githubusercontent.com/119160414/236832465-3bea3445-beaf-405e-bffa-912eeed947cb.png)
## Which City has Highest profit?
![1](https://user-images.githubusercontent.com/119160414/236832527-70da59cd-3596-479f-aebc-6de5ca0b74ff.png)

![2](https://user-images.githubusercontent.com/119160414/236832555-280b325b-947b-4189-8716-98e8c62725ba.png)
## Which ship mode is profitable?
![1](https://user-images.githubusercontent.com/119160414/236832640-856b167c-468b-4849-862c-40e5d30fde9d.png)

![2](https://user-images.githubusercontent.com/119160414/236832604-291704cb-4daf-4047-8d4e-c015de9c001b.png)

![3](https://user-images.githubusercontent.com/119160414/236832832-e1865df3-e702-4ff0-a46b-7035d46ddcd7.png)

![4](https://user-images.githubusercontent.com/119160414/236832903-a07c5b85-e6b6-4843-b1d8-cd2401e4348f.png)
## Sales of the product based on region.
![1](https://user-images.githubusercontent.com/119160414/236832655-44d2b6ae-5d42-4622-b919-1fa5f3bae582.png)

![2](https://user-images.githubusercontent.com/119160414/236832760-2a2b3865-8fb4-48b0-a912-d7e3928dc216.png)
## Find the relation between sales and profit
![1](https://user-images.githubusercontent.com/119160414/236832670-bfddd4fe-82af-4969-94c5-077dab1fce87.png)

![3](https://user-images.githubusercontent.com/119160414/236832844-de9be282-73c6-44a0-bbd9-770d31189c46.png)

![2](https://user-images.githubusercontent.com/119160414/236832766-28e1f709-caf1-46fa-be2a-e6c222d4b7bd.png)

## Heatmap
![heat](https://user-images.githubusercontent.com/119160414/236833028-b7386e46-f9b1-4653-b136-ec1dbcb42197.png)

## Find the relation between sales and profit based on the following category.

### Segment
![1](https://user-images.githubusercontent.com/119160414/236832694-fbb537c1-3724-4ea1-b770-4e445cd4a722.png)
### City
![2](https://user-images.githubusercontent.com/119160414/236832784-e06a0575-4f16-433d-b41b-1f28d72c36a5.png)
### States
![3](https://user-images.githubusercontent.com/119160414/236832855-c6c13bb5-1cad-4659-9045-f327b1e8b50d.png)
### Segment and Ship Mode
![4](https://user-images.githubusercontent.com/119160414/236832919-210be8a3-75fe-4594-9152-f027cce4a17b.png)
### Segment, Ship mode and Region
![5](https://user-images.githubusercontent.com/119160414/236832960-9603bdff-8250-4908-9c7f-d62bc0d75c0a.png)

# Result:
Thus, Data Visualization is performed on the given dataset and save the data to a file.
