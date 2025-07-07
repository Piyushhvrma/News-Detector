import pandas as pd

# Load the individual CSVs
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels: Fake = 0, True = 1
df_fake["label"] = 0
df_true["label"] = 1

# Combine them
df = pd.concat([df_fake, df_true], ignore_index=True)

# Only keep text and label columns
df = df[["text", "label"]]

# Save to a single clean CSV file
df.to_csv("news.csv", index=False)

print("✅ Successfully created news.csv")
