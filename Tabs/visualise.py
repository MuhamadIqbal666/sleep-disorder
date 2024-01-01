import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # Import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Fungsi untuk membuat heatmap confusion matrix
def plot_confusion_matrix_heatmap(y_true, y_pred, title=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))  # Adjust the size of the plot here
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])

    # Set the title if provided
    if title:
        plt.title(title)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()

# Fungsi untuk melatih model Decision Tree
def train_model(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi Sleep Disorder")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Checkbox untuk Plot Confusion Matrix
    if st.checkbox("Plot Confusion Matrix"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        plot_confusion_matrix_heatmap(y_test, y_pred)

    # Checkbox untuk Plot Decision Tree
    if st.checkbox("Plot Decision Tree"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model = train_model(x_train, y_train)
        dot_data = export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['none', 'sleep apnea', 'insomnia']
        )

        st.graphviz_chart(dot_data)

# Panggil fungsi utama aplikasi
if __name__ == '__main__':
    # Assuming these are the columns you want to include
    columns_to_include = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps']

    # Select only the columns you want from the DataFrame
    df = pd.DataFrame()
    x = df[columns_to_include]
    y = df['Sleep_Disorder']

    app(df, x, y)
