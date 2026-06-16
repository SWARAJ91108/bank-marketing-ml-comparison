#Predict whether a bank customer will subscribe to a term deposit based on their demographic information, financial details
#and previous marketing campaign interactions 
# using supervised machine learning algorithms. 

import pandas as pd
from sklearn.metrics import accuracy_score
df = pd.read_csv("bank_marketing_rows.csv")




print(df.dtypes)

#label encoding Convert categorical (text) columns into numerical values
## Label Encoding: Convert categorical (text) columns into numerical values
# so that machine learning algorithms can process them.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])




#feature selection
X = df.drop("subscribe",axis=1)
y = df["subscribe"]




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#Scaling 
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train_scaled,y_train)
LR_pred = LR.predict(X_test_scaled)
LR_acc = accuracy_score(y_test,LR_pred)



#Knn
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train_scaled,y_train)
KNN_pred = KNN.predict(X_test_scaled)
KNN_acc = accuracy_score(y_test,KNN_pred)


#Naive bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train_scaled,y_train)
NB_pred = NB.predict(X_test_scaled)
NB_acc = accuracy_score(y_test,NB_pred)

#SVM Support vector Machine
from sklearn.svm import SVC
SVC1 = SVC()
SVC1.fit(X_train_scaled,y_train)
SVC_pred = SVC1.predict(X_test_scaled)
SVC_acc = accuracy_score(SVC_pred,y_test)


#Decsion tree 
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train_scaled,y_train)
DT_pred = DT.predict(X_test_scaled)
DT_acc = accuracy_score(y_test,DT_pred)

#Random forest 
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100,random_state=42)
RF.fit(X_train_scaled,y_train)
RF_pred = RF.predict(X_test_scaled)
RF_acc = accuracy_score(y_test,RF_pred)

#ADA BOOST
from sklearn.ensemble import AdaBoostClassifier
ADA = AdaBoostClassifier(n_estimators=50,random_state=42)
ADA.fit(X_train,y_train)
ADA_pred = ADA.predict(X_test)
ADA_acc = accuracy_score(y_test,ADA_pred)

#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(
    n_estimators=50,
    random_state= 42
)
GB.fit(X_train,y_train)
GB_pred = GB.predict(X_test)
GB_acc = accuracy_score(y_test,GB_pred)

#XGBOOSTING
from xgboost import XGBClassifier
XGB = XGBClassifier(random_state=42)
XGB.fit(X_train,y_train)
XGB_pred = XGB.predict(X_test)
XGB_acc = accuracy_score(y_test,XGB_pred)





#Stacking
LR_train_pred = LR.predict(X_train_scaled)
KNN_train_pred = KNN.predict(X_train_scaled)
RF_train_pred = RF.predict(X_train_scaled)


meta_X_train= pd.DataFrame({
    'LR' :LR_train_pred,
    'KNN':KNN_train_pred,
    'RF' :RF_train_pred
}
)

meta_X_test = pd.DataFrame({
    'LR':LR_pred,
    'KNN':KNN_pred,
    'RF':RF_pred
})

from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()
meta_model.fit(meta_X_train,y_train)
stack_pred = meta_model.predict(meta_X_test)
Stack_acc = accuracy_score(y_test,stack_pred)





#Plotting 
# Store all accuracies

results = {
    "Logistic Regression": LR_acc,
    "KNN": KNN_acc,
    "Naive Bayes": NB_acc,
    "SVM": SVC_acc,
    "Decision Tree": DT_acc,
    "Random Forest": RF_acc,
    "AdaBoost": ADA_acc,
    "Gradient Boosting": GB_acc,
    "XGBoost": XGB_acc,
    "Stacking": Stack_acc
}

# Plot comparison

import matplotlib.pyplot as plt
import pandas as pd

comparison_df = pd.DataFrame({
    "Algorithm": results.keys(),
    "Accuracy": results.values()
})

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

print(comparison_df)

plt.figure(figsize=(12,6))

bars = plt.bar(
    comparison_df["Algorithm"],
    comparison_df["Accuracy"]
)

plt.title("Comparison of Supervised Learning Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)

# Display accuracy values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center'
    )

plt.tight_layout()
plt.show()



