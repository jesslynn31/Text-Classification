import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


df = pd.read_csv("rotten_tomatoes_predictions.csv")

def accuracy_pie_chart():
    correct_counts = df['correct'].value_counts()
    plt.pie(correct_counts, labels=["Incorrect", "Correct"], autopct='%1.1f%%', startangle=140)
    plt.title("Model Prediction Accuracy")
    plt.axis('equal')
    plt.show()

def sk_confusion_matrix():
    cm = confusion_matrix(df['true_label'], df['predicted_label'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    sk_confusion_matrix()
    accuracy_pie_chart()


if __name__ == "__main__":
    main()