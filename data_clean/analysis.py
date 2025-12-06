import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="whitegrid" palette="pastel")
import numpy as np

df=pd.read_csv(r"C:\Users\LENOVO LOQ\Desktop\DSproject\ecommerce-customer-segmentation\data_clean\cleaned_purchase_data_exe.csv")

print(df.head())
print(df.info())
print(df.describe())

# Total sales value per product category
category_sales = df.groupby('product_category')['value [USD]'].sum().sort_values(ascending=False)

# Display results
print(category_sales)


# Top 10 customers by total sales value
customer_sales = df.groupby('customer_id')['value [USD]'].sum().sort_values(ascending=False)
print(customer_sales.head(10))


# Crosstab of product category vs payment method
pd.crosstab(df['product_category'], df['payment_method'], values=df['value [USD]'], aggfunc='sum').plot(kind='bar', stacked=True)
plt.title('Payment Methods Used per Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.show()

#how most users behave(Fast buyers or long browsers)
sns.histplot(df['time_on_site [Minutes]'], bins=20, kde=True)
plt.title('Distribution of Time Spent on Site')
plt.xlabel('Time on Site (minutes)')
plt.show()

sns.histplot(df['clicks_in_site'], bins=20, kde=True)
plt.title('Distribution of Clicks in Site')
plt.xlabel('Clicks')
plt.show()



# Visualization
plt.figure(figsize=(8,5))
category_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Value')
plt.show()


# Count and total value by payment method
payment_stats = df.groupby('payment_method')['value [USD]'].agg(['count', 'sum'])
print(payment_stats)

# Pie chart for payment method distribution
df['payment_method'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
plt.title('Payment Method Distribution')
plt.ylabel('')
plt.show()

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group by date to get total daily sales
daily_sales = df.groupby('date')['value [USD]'].sum()

# Plot daily sales trend
plt.figure(figsize=(10,5))
daily_sales.plot(kind='line', marker='o')
plt.title('Daily Sales Trend (November 2018)')
plt.xlabel('Date')
plt.ylabel('Total Sales Value')
plt.grid(True)
plt.show()

# Scatter plot - time on site vs value
plt.figure(figsize=(8,5))
plt.scatter(df['time_on_site [Minutes]'], df['value [USD]'], alpha=0.6, color='purple')
plt.title('Time on Site vs Cart Value')
plt.xlabel('Time on Site (minutes)')
plt.ylabel('Cart Value')
plt.show()

# Scatter plot - clicks vs value
plt.figure(figsize=(8,5))
plt.scatter(df['clicks_in_site'], df['value [USD]'], alpha=0.6, color='teal')
plt.title('Clicks on Site vs Cart Value')
plt.xlabel('Number of Clicks')
plt.ylabel('Cart Value')
plt.show()


# Correlation between numeric columns
corr = df[['value [USD]', 'time_on_site [Minutes]', 'clicks_in_site']].corr()

# Display correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Numeric Features')
plt.show()

