import pandas as pd

import pandas as pd

pd = pd.read_csv(r"C:\Users\iyehi\OneDrive\Desktop\DS grp pro\ecommerce-customer-segmentation\data_raw\purchase_data_exe.csv")

print(pd.head())
print(pd.info())
print(pd.describe())
print(pd.isnull().sum())
print(pd.duplicated().sum())
pd = pd.drop_duplicates()
print(pd.duplicated().sum())
pd.to_csv(r"C:\Users\iyehi\OneDrive\Desktop\DS grp pro\ecommerce-customer-segmentation\data_raw\cleaned_purchase_data_exe.csv", index=False)



