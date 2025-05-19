import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

zip_path = 'smoking-drinking-dataset.zip'
csv_filename = 'smoking_driking_dataset_Ver01.csv'


with ZipFile(zip_path) as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

df['DRK_YN'] = df['DRK_YN'].map({'Y': 1, 'N': 0})
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

X = df.drop(columns=['DRK_YN'])
y = df['DRK_YN']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_val_norm = (X_val - X_train.min()) / (X_train.max() - X_train.min())
X_test_norm = (X_test - X_train.min()) / (X_train.max() - X_train.min())

train_df = X_train_norm.copy()
train_df['DRK_YN'] = y_train.values

val_df = X_val_norm.copy()
val_df['DRK_YN'] = y_val.values

test_df = X_test_norm.copy()
test_df['DRK_YN'] = y_test.values

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)