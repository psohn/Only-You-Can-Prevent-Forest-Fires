### because how can you not use random forest regression on a forest fire
### regression? this is lazily coded and uncommented. purely for entertainment
### and completion

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

train_X = pd.read_csv('data/features_train.csv', index_col = 0)
train_y = pd.read_csv('data/target_train.csv', index_col = 0).astype(int).values.ravel()

test_X = pd.read_csv('data/features_test.csv', index_col = 0)
test_y = pd.read_csv('data/target_test.csv', index_col = 0).astype(int).values.ravel()

model = RandomForestClassifier(random_state = 0, max_depth = 5).fit(X = train_X, y = train_y)

pred_train = model.predict(train_X)
pred_test = model.predict(test_X)

r2_train = r2_score(train_y, pred_train)
r2_test = r2_score(test_y, pred_test)

mse_train = mean_squared_error(train_y, pred_train, squared = False)
mse_test = mean_squared_error(test_y, pred_test, squared = False)

print(r2_train, mse_train)
print(r2_test, mse_test)

### r2_train = -0.1386
### mse_train = 0.5807

### r2_test = -0.1518
### mse_test = 0.6487