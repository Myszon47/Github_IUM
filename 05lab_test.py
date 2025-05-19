import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = tf.keras.models.load_model("myModel.h5", compile=False)

test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=["DRK_YN"])
y_test = test_df["DRK_YN"]

predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()

output_df = pd.DataFrame({
    "expected": y_test,        
    "predicted": predicted_classes 
})

accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes)

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

output_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
print("Metrics saved to metrics.txt")