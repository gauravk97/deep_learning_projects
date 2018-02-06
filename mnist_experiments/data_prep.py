import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[train.columns[1:]], train[train.columns[0]].values, test_size=0.20, random_state=42, stratify=train[train.columns[1]].values)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train["labels"] = y_train
X_test["labels"]= y_test
X_train.to_csv("data_train.csv", index=False)
X_test.to_csv("data_val.csv", index=False)