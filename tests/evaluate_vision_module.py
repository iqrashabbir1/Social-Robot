import pandas as pd

csv_path = "tests/emotion_log_labeled.csv"

df = pd.read_csv(csv_path)

assert "emotion" in df.columns
assert "true_emotion" in df.columns

labels = sorted(df["true_emotion"].unique())

# Initialize confusion matrix
conf_mat = {t: {p: 0 for p in labels} for t in labels}

for _, row in df.iterrows():
    t = row["true_emotion"]
    p = row["emotion"]
    if p not in labels:
        continue
    conf_mat[t][p] += 1

print("Confusion Matrix (rows = true, cols = predicted):")
print("          " + "  ".join(f"{p:>8}" for p in labels))

correct = 0
total = 0
for t in labels:
    row_counts = []
    for p in labels:
        c = conf_mat[t][p]
        row_counts.append(c)
        total += c
        if t == p:
            correct += c
    row_str = "  ".join(f"{c:>8}" for c in row_counts)
    print(f"{t:>8}  {row_str}")

accuracy = correct / total if total > 0 else 0.0
print(f"\nOverall accuracy: {accuracy*100:.2f}%")
