import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Import plotting utility (if needed)
from svm_utils import plot_decision_boundary

def main():
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels

    # For visualization purposes, we use only the first two features.
    X_vis = X[:, :2]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_vis_train, X_vis_test, _, _ = train_test_split(X_vis, y, test_size=0.3, random_state=42)

    # Initialize and train the SVM classifier (linear kernel)
    clf = SVC(kernel='linear', C=1, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Plot decision boundary using the first two features (for visualization)
    plot_decision_boundary(clf, X_vis_train, y_train, title="SVM Decision Boundary (Iris Dataset)")

if __name__ == "__main__":
    main()
