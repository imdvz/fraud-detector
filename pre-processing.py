import pandas as pd
import numpy as np
from sklearn import preprocessing

# Read train and test data
df_train = pd.read_csv("fraudTrain.csv")
df_test = pd.read_csv("fraudTest.csv")

df_train = df_train.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df_train = df_train.loc[df_train['is_fraud'] == 1]
non_fraud_df_train = df_train.loc[df_train['is_fraud'] == 0][:7506]

normal_distributed_df_train = pd.concat([fraud_df_train, non_fraud_df_train])

# Shuffle dataframe rows
new_df_train = normal_distributed_df_train.sample(frac=1, random_state=42)

df_test = df_test.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df_test = df_test.loc[df_test['is_fraud'] == 1]
non_fraud_df_test = df_test.loc[df_test['is_fraud'] == 0][:2145]

normal_distributed_df_test = pd.concat([fraud_df_test, non_fraud_df_test])

# Shuffle dataframe rows
new_df_test = normal_distributed_df_test.sample(frac=1, random_state=42)

df = pd.concat([new_df_train, new_df_test])
df_backup = df

  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# List of columns for label encoding
colmns = ['category', 'gender', 'city', 'state', 'job']

for column in colmns:
    # Encode labels in column
    df[column]= label_encoder.fit_transform(df[column])

# Making age of user using 'dob' column
df['dob'] = df['dob'].str[0:4]
df['dob'] = df['dob'].astype(int)
df['dob'] = 2022-df['dob']

# Making transaction_hour column
df['transaction_hour'] = df['trans_date_trans_time'].str[11:13]
df['transaction_hour'] = df['transaction_hour'].astype(int)

# Making column for weekday on which transaction was done
df["transaction_date"] = df['trans_date_trans_time'].str[:10]
df["transaction_date"] = df["transaction_date"].apply(pd.to_datetime)
# The day of the week with Monday=0, Sunday=6.
df["transaction_weekday"] = df["transaction_date"].dt.dayofweek

# Dropping columns
df1 = df.drop(['trans_date_trans_time', 'merchant', 'Unnamed: 0', 'trans_num', 'first', 'last',
              'unix_time', 'transaction_date'], axis='columns')

df_final = df1[['category', 'amt', 'gender', 'dob', 'transaction_hour', 'is_fraud']]
df_final = df_final.sample(frac=1, random_state=0)

X = df_final[['category', 'amt', 'gender', 'dob', 'transaction_hour']]
y = df_final[['is_fraud']]

# Save pre-processed dataset
df_final.to_csv("cleaned_fraud_data.csv", index=False)