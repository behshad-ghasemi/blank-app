import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
)

data = r"C:\Users\behsh\Desktop\CARDEA\tetrad new data python\mtDNA\transformed_data0 MtDNA.xlsx"
# Load the Excel file" 
df = pd.read_excel( data )
print(df.head())
X = df.drop(columns=['Heart Failure /control'])  
y = df['Heart Failure /control']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)   


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1️⃣ Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)


mse_log = mean_squared_error(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
conf_matrix_log = confusion_matrix(y_test, y_pred_log)

log_coeff = log_model.coef_[0]

### 2️⃣ Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Feature Importance
rf_importances = rf_model.feature_importances_

### 3️⃣ Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
acc_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)

# Feature Importance
gb_importances = gb_model.feature_importances_

### 4️⃣ Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Feature Importance
dt_importances = dt_model.feature_importances_

print("\n==== Logistic Regression ====")
print(f"Coefficients: {log_coeff}")
print(f"MSE: {mse_log}, Accuracy: {acc_log}, Precision: {precision_log}, Recall: {recall_log}")
print(f"Confusion Matrix:\n{conf_matrix_log}")

print("\n==== Random Forest ====")
print(f"Feature Importance: {rf_importances}")
print(f"MSE: {mse_rf}, Accuracy: {acc_rf}, Precision: {precision_rf}, Recall: {recall_rf}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")

print("\n==== Gradient Boosting ====")
print(f"Feature Importance: {gb_importances}")
print(f"MSE: {mse_gb}, Accuracy: {acc_gb}, Precision: {precision_gb}, Recall: {recall_gb}")
print(f"Confusion Matrix:\n{conf_matrix_gb}")

print("\n==== Decision Tree ====")
print(f"Feature Importance: {dt_importances}")
print(f"MSE: {mse_dt}, Accuracy: {acc_dt}, Precision: {precision_dt}, Recall: {recall_dt}")
print(f"Confusion Matrix:\n{conf_matrix_dt}")


plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.barplot(x=X.columns, y=rf_importances)
plt.xticks(rotation=90)
plt.title("Random Forest - Feature Importance")


plt.subplot(1, 3, 2)
sns.barplot(x=X.columns, y=gb_importances)
plt.xticks(rotation=90)
plt.title("Gradient Boosting - Feature Importance")

plt.subplot(1, 3, 3)
sns.barplot(x=X.columns, y=dt_importances)
plt.xticks(rotation=90)
plt.title("Decision Tree - Feature Importance")

plt.show()
best_log_model = LogisticRegression(C=5, penalty='l1', solver='liblinear')
best_log_model.fit(X_train_scaled, y_train)
y_pred_log_best = best_log_model.predict(X_test_scaled)

acc_log_best = accuracy_score(y_test, y_pred_log_best)
precision_log_best = precision_score(y_test, y_pred_log_best)
recall_log_best = recall_score(y_test, y_pred_log_best)
conf_matrix_log_best = confusion_matrix(y_test, y_pred_log_best)

print("\n==== Optimized Logistic Regression ====")
print(f"Accuracy: {acc_log_best}, Precision: {precision_log_best}, Recall: {recall_log_best}")
print(f"Confusion Matrix:\n{conf_matrix_log_best}")

best_rf_model = RandomForestClassifier(
    n_estimators=150, min_samples_split=5, min_samples_leaf=1,
    max_features='log2', max_depth=2, random_state=42
)
best_rf_model.fit(X_train, y_train)
y_pred_rf_best = best_rf_model.predict(X_test)

acc_rf_best = accuracy_score(y_test, y_pred_rf_best)
precision_rf_best = precision_score(y_test, y_pred_rf_best)
recall_rf_best = recall_score(y_test, y_pred_rf_best)
conf_matrix_rf_best = confusion_matrix(y_test, y_pred_rf_best)

print("\n==== Optimized Random Forest ====")
print(f"Accuracy: {acc_rf_best}, Precision: {precision_rf_best}, Recall: {recall_rf_best}")
print(f"Confusion Matrix:\n{conf_matrix_rf_best}")


best_gb_model = GradientBoostingClassifier(
    learning_rate=0.01, max_depth=1, n_estimators=200, random_state=42
)
best_gb_model.fit(X_train, y_train)
y_pred_gb_best = best_gb_model.predict(X_test)

acc_gb_best = accuracy_score(y_test, y_pred_gb_best)
precision_gb_best = precision_score(y_test, y_pred_gb_best)
recall_gb_best = recall_score(y_test, y_pred_gb_best)
conf_matrix_gb_best = confusion_matrix(y_test, y_pred_gb_best)

print("\n==== Optimized Gradient Boosting ====")
print(f"Accuracy: {acc_gb_best}, Precision: {precision_gb_best}, Recall: {recall_gb_best}")
print(f"Confusion Matrix:\n{conf_matrix_gb_best}")
import shap
#shap
print('SHAP Values for Logistic Regression')
explainer_log = shap.Explainer(best_log_model, X_train_scaled)
shap_values_log = explainer_log(X_test_scaled)
shap.summary_plot(shap_values_log, X_test, plot_type="bar")
shap.summary_plot(shap_values_log, X_test)


print('SHAP Values for  Gradient Boosting')
explainer_gb = shap.Explainer(best_gb_model, X_train)
shap_values_gb = explainer_gb(X_test)
shap.summary_plot(shap_values_gb, X_test, plot_type="bar")
shap.summary_plot(shap_values_gb, X_test)


for feature in X.columns:
    print(f"Plotting SHAP dependence for feature: {feature}")
    shap.dependence_plot(feature, shap_values_gb.values, X_test, feature_names=X.columns)

#new patient

feature_names = ['mtDNA cn'	,'telomere ', 	'miR-21',	'miR-92'] 
new_patient = np.array([[ '0.113447927'	,'1'	,'0.012179888'	,'0.032066508']])
new_patient_df = pd.DataFrame(new_patient, columns=feature_names)
new_patient_scaled = scaler.transform(new_patient_df)

#probability for each model
prob_log = best_log_model.predict_proba(new_patient_scaled)[:, 1]  
prob_rf = best_rf_model.predict_proba(new_patient_df)[:, 1]
prob_gb = best_gb_model.predict_proba(new_patient_df)[:, 1]
prob_dt = dt_model.predict_proba(new_patient_df)[:, 1]


print(f"🔹 Logistic Regression - Probability of getting HfpEf: {prob_log[0]:.4f}")
print(f"🔹 Random Forest - Probability of getting HfpEf: {prob_rf[0]:.4f}")
print(f"🔹 Gradient Boosting - Probability of getting HfpEf: {prob_gb[0]:.4f}")
print(f"🔹 Decision Tree - Probability of getting HfpEf: {prob_dt[0]:.4f}")

# threshold
threshold = 0.5
pred_log = (prob_log >= threshold).astype(int)
print(f'===🔹 Logistic Regression - final result ------> {'heart failure' if pred_log[0] == 1 else 'healthy'}')
print(f'===🔹 Random Forest - final result ------> {'heart failure' if pred_log[0] == 1 else 'healthy'}')
print(f"===🔹 Gradient Boosting - final result ------> {'heart failure' if pred_log[0] == 1 else 'healthy'}")
print(f"===🔹 Decision Tree - final result ------> {'heart failure' if pred_log[0] == 1 else 'healthy'}")
### 1 Logistic Regression
y_pred_Bestlog = best_log_model.predict(X_test_scaled)
BEST_mse_log = mean_squared_error(y_test, y_pred_Bestlog)
BEST_acc_log = accuracy_score(y_test, y_pred_Bestlog)
BEST_precision_log = precision_score(y_test, y_pred_Bestlog)
BEST_recall_log = recall_score(y_test, y_pred_Bestlog)
BEST_conf_matrix_log = confusion_matrix(y_test, y_pred_Bestlog)

BEST_log_coeff = best_log_model.coef_[0]

### 2 Random Forest
y_pred_Best_rf = best_rf_model.predict(X_test)
BEST_mse_rf = mean_squared_error(y_test, y_pred_Best_rf)
BEST_acc_rf = accuracy_score(y_test, y_pred_Best_rf)
BEST_precision_rf = precision_score(y_test, y_pred_Best_rf)
BEST_recall_rf = recall_score(y_test, y_pred_Best_rf)
BEST_conf_matrix_rf = confusion_matrix(y_test, y_pred_Best_rf)

# Feature Importance
BEST_rf_importances = best_rf_model.feature_importances_

### 3️⃣ Gradient Boosting
y_pred_Best_gb = best_gb_model.predict(X_test)

BEST_mse_gb = mean_squared_error(y_test, y_pred_Best_gb)
BEST_acc_gb = accuracy_score(y_test, y_pred_Best_gb)
BEST_precision_gb = precision_score(y_test, y_pred_Best_gb)
BEST_recall_gb = recall_score(y_test, y_pred_Best_gb)
BEST_conf_matrix_gb = confusion_matrix(y_test, y_pred_Best_gb)

# Feature Importance
BEST_gb_importances = best_gb_model.feature_importances_


print("\n==== Logistic Regression ====")
print(f"Coefficients: {BEST_log_coeff}")
print(f"MSE: {BEST_mse_log}, Accuracy: {BEST_acc_log}, Precision: {BEST_precision_log}, Recall: {BEST_recall_log}")
print(f"Confusion Matrix:\n{BEST_conf_matrix_log}")

print("\n==== Random Forest ====")
print(f"Feature Importance: {BEST_rf_importances}")
print(f"MSE: {BEST_mse_rf}, Accuracy: {BEST_acc_rf}, Precision: {BEST_precision_rf}, Recall: {BEST_recall_rf}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")

print("\n==== Gradient Boosting ====")
print(f"Feature Importance: {BEST_gb_importances}")
print(f"MSE: {BEST_mse_gb}, Accuracy: {BEST_acc_gb}, Precision: {BEST_precision_gb}, Recall: {BEST_recall_gb}")
print(f"Confusion Matrix:\n{BEST_conf_matrix_gb}")


plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.barplot(x=X.columns, y=BEST_rf_importances)
plt.xticks(rotation=90)
plt.title("Random Forest - Feature Importance")


plt.subplot(1, 3, 2)
sns.barplot(x=X.columns, y=BEST_gb_importances)
plt.xticks(rotation=90)
plt.title("Gradient Boosting - Feature Importance")


plt.show()
