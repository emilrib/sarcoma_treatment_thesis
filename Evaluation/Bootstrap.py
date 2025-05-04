import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from scipy.stats import mode
from global_config import datasets_dir
from sklearn.metrics import accuracy_score

df_path = os.path.join(datasets_dir, "df_with_merged_groups.csv")
df = pd.read_csv(df_path)
df = df.dropna(subset=['chemo_status', 'survival_status'])

X = df.drop(columns=['Pat ID', 'survival_status', 'chemo_status'])
y = df['survival_status'].astype(int)

# ---------------------------
# Encode categorical if necessary
# ---------------------------
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# Bagged Trees with OOB and Test Error
# ---------------------------
B = 100
n = X_train.shape[0]
oob_errors = []
test_errors = []
all_preds = np.zeros((B, n))
oob_indices_list = []

for b in range(B):
    indices = np.random.choice(n, size=n, replace=True)
    oob_indices = list(set(range(n)) - set(indices))
    oob_indices_list.append(oob_indices)

    X_bootstrap = X_train.iloc[indices]
    y_bootstrap = y_train.iloc[indices]

    model = DecisionTreeClassifier(random_state=b)
    model.fit(X_bootstrap, y_bootstrap)

    # Store OOB predictions
    preds = model.predict(X_train)
    all_preds[b, :] = preds

    # Compute OOB error for this round
    oob_preds = np.zeros(n)
    oob_counts = np.zeros(n)
    for i in oob_indices:
        oob_preds[i] += preds[i]
        oob_counts[i] += 1

    oob_final = []
    for i in range(n):
        if oob_counts[i] > 0:
            pred_vals = [int(all_preds[j, i]) for j in range(b + 1) if i in oob_indices_list[j]]
            if pred_vals:
                majority_vote = mode(pred_vals, keepdims=True).mode[0]  # Explicitly handle the result of mode
            else:
                majority_vote = y_train.iloc[i]  # Fallback if pred_vals is empty

            oob_final.append((majority_vote, y_train.iloc[i]))

    # Final calculation for OOB error
    if oob_final:  # Avoid errors if oob_final is empty
        oob_error = 1 - np.mean([int(p == t) for p, t in oob_final])
    else:
        oob_error = 0  # Default to 0 error if no OOB predictions

    oob_errors.append(oob_error)  # Append the OOB error to the list

    # Compute test error
    test_preds = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_preds)
    test_errors.append(test_error)


# ---------------------------
# Plot OOB vs Test Error
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(range(1, B+1), oob_errors, label='OOB Error', marker='o')
plt.plot(range(1, B+1), test_errors, label='Test Error', marker='x')
plt.xticks(np.arange(0, B+1, 25))
plt.xlabel("Bootstrap Iteration")
plt.ylabel("Error Rate")
plt.title("OOB vs Test Error of Bagged Tree")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()