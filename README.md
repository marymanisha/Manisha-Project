import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Superstore_Sales_Dataset.csv')
df
df.info()
df.head()
df.interpolate(method='linear', inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Week'] = df['Order Date'].dt.isocalendar().week
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['Is_Payday'] = df['Day'].isin([15, df['Order Date'].dt.daysinmonth])
df['Is_Earthquake'] = df['Order Date'] == pd.Timestamp('2016-04-16')
df['Sales_Rolling_7'] = df['Sales Amount'].rolling(window=7).mean()
df['Sales_Rolling_30'] = df['Sales Amount'].rolling(window=30).mean()
df['Sales_Lag_7'] = df['Sales Amount'].shift(7)
df['Sales_Lag_30'] = df['Sales Amount'].shift(30)
df['Avg_Sales_Category'] = df.groupby('Category')['Sales Amount'].transform('mean')
df['Avg_Sales_Sub-Category'] = df.groupby('Sub-Category')['Sales Amount'].transform('mean')
top_products = df.groupby('Sub-Category')['Sales Amount'].sum().nlargest(5)
print("Top-Selling Products:", top_products)
plt.figure(figsize=(14,6))
df.groupby('Order Date')['Sales Amount'].sum().plot()
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()
corr = df[['Sales Amount', 'Profit', 'Quantity']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
