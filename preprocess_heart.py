import pandas as pd

df = pd.read_csv("heart.csv")

# Encode Sex
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

# Encode Chest Pain Type
df["ChestPainType"] = df["ChestPainType"].map({
    "ATA": 0,
    "NAP": 1,
    "ASY": 2,
    "TA": 3
})

# Encode RestingECG
df["RestingECG"] = df["RestingECG"].map({
    "Normal": 0,
    "ST": 1,
    "LVH": 2
})

# Encode Exercise Angina
df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})

# Encode ST Slope
df["ST_Slope"] = df["ST_Slope"].map({
    "Up": 0,
    "Flat": 1,
    "Down": 2
})

df.to_csv("processed_heart.csv", index=False)
print("Saved processed_heart.csv")
