import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_generator):
    test_generator.reset()
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    Y_pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_generator.classes, y_pred)
    class_names = list(test_generator.class_indices.keys())

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(test_generator.classes, y_pred, target_names=class_names))
    return test_accuracy
