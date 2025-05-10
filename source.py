
# 1. Upload the Dataset
from google.colab import files
uploaded = files.upload()

# 2. Load the Dataset
import pandas as pd
df = pd.read_csv(next(iter(uploaded)))
df.head()

# 3. Data Exploration
df.info()
df.describe()

# 4. Check for Missing Values and Duplicates
print("Missing values per column:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# 5. Visualize a Few Features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(y='Disease', data=df, order=df['Disease'].value_counts().index[:10])
plt.title('Top 10 Most Frequent Diseases')
plt.show()

# 6. Identify Target and Features
target = 'Disease'
features = [col for col in df.columns if col != target]

# 7. Convert Categorical Columns to Numerical (Fill NaNs first)
df.fillna("None", inplace=True)

# 8. One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=features)

# 9. Feature Scaling (Optional for Tree-Based Models)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df_encoded.drop(target, axis=1)
X_scaled = scaler.fit_transform(X)
y = df_encoded[target]

# 10. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11. Model Building
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 12. Evaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 13. Make Predictions from New Input
# Example input with dummy encoded values (you can adapt this to match actual encoded format)
import numpy as np
sample_input = np.zeros((1, X.shape[1]))  # All zeros
sample_prediction = model.predict(sample_input)

# 14. Convert to DataFrame and Encode (Handled earlier with get_dummies)
# Assuming new input is in raw format, convert it to same format as training data

# 15. Predict the Final Grade (If applicable to disease severity; placeholder here)
# print("Predicted Grade/Severity:", sample_prediction)

# 16. Deployment - Building an Interactive App
!pip install gradio

# 17. Create a Prediction Function
import gradio as gr

def predict_disease(**inputs):
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)
    return f"Predicted Disease: {pred[0]}"

# 18. Create the Gradio Interface
input_components = [gr.Textbox(label=col) for col in features[:5]]  # Adjust number of features
gr.Interface(fn=predict_disease, inputs=input_components, outputs="text").launch()
