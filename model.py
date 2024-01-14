import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# dataset
df=pd.read_csv("heart_2020_cleaned.csv")

# removing duplicate
duplicateObser = df[df.duplicated()]
LabelsDupObser=duplicateObser.axes[0].tolist()
print('Number of duplicated observations:', duplicateObser.shape[0])
df=df.drop_duplicates()
print(df.shape)

#data maping
df.replace("Yes",1,inplace=True)
df.replace("No",0,inplace=True)

target=df["HeartDisease"]
df.drop(["HeartDisease"], axis=1, inplace=True)
df.AgeCategory.unique()
df.replace("18-24",0,inplace=True)
df.replace("25-29",1,inplace=True)
df.replace("30-34",2,inplace=True)
df.replace("35-39",3,inplace=True)
df.replace("40-44",4,inplace=True)
df.replace("45-49",5,inplace=True)
df.replace("50-54",6,inplace=True)
df.replace("55-59",7,inplace=True)
df.replace("60-64",8,inplace=True)
df.replace("65-69",9,inplace=True)
df.replace("70-74",10,inplace=True)
df.replace("75-79",11,inplace=True)
df.replace("80 or older",13,inplace=True)

df.Diabetic.unique()
df.replace("No, borderline diabetes",2,inplace=True)
df.replace("Yes (during pregnancy)",3,inplace=True)

df.GenHealth.unique()
df.replace("Excellent",0,inplace=True)
df.replace("Good",1,inplace=True)
df.replace("Fair",2,inplace=True)
df.replace("Very good",3,inplace=True)
df.replace("Poor",4,inplace=True)

df.Race.unique()
df.replace("White",0,inplace=True)
df.replace("Other",1,inplace=True)
df.replace("Black",2,inplace=True)
df.replace("Hispanic",3,inplace=True)
df.replace("Asian",4,inplace=True)
df.replace("American Indian/Alaskan Native",4,inplace=True)

df.Sex.unique()
df.replace("Female",0,inplace=True)
df.replace("Male",1,inplace=True)

df['BMI'].mask(df['BMI']  < 18.5, 0, inplace=True)
df['BMI'].mask(df['BMI'].between(18.5,25), 1, inplace=True)
df['BMI'].mask(df['BMI'].between(25,30), 2, inplace=True)
df['BMI'].mask(df['BMI']  > 30, 3, inplace=True)

# Split the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(df, target, test_size=1000, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a logistic regression model on the training set
LogRegModel=LogisticRegression()
LogRegModel.fit(X_train, y_train)

y_pred = LogRegModel.predict(X_test)
from sklearn.metrics import accuracy_score
LogRegModelAccuraccy = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy: {:.2f}%".format(LogRegModelAccuraccy * 100))
# Save the model using pickle
with open('LogRegModel.pkl', 'wb') as f:
    pickle.dump(LogRegModel, f)

#Train a knn model on the training set
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN Accuracy:{:.2f}%".format(knn.score(X_test, y_test)*100))

with open('knn.pkl', 'wb') as f1:
    pickle.dump(knn, f1)


#Train a XGB model on the the training set
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Convert target variable to numerical values using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train_encoded)

# Predictions on the test set
y_pred_encoded = xgb.predict(X_test)
# Decode predictions back to original labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("XGB Accuracy: {:.2f}%".format(accuracy * 100))

with open('xgb.pkl', 'wb') as f2:
    pickle.dump(xgb, f2)



#Nural network model
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions (0 or 1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

with open('NN.pkl', 'wb') as f3:
    pickle.dump(model, f3)



#GaussianNB model:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Standardize the data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = nb_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Classification report (includes precision, recall, and F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

with open('GNB.pkl', 'wb') as f4:
    pickle.dump(nb_model, f4)

