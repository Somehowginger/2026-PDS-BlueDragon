import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

metadata_csv = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"
annotations_csv = Path(__file__).resolve().parent.parent / "data" / "annotations_combined.csv"
metadata_df = pd.read_csv(metadata_csv)
annotations_df = pd.read_csv(annotations_csv)
df = metadata_df.merge(annotations_df, on='img_id', how='inner')

# Number of data columns in metadata
print("Number of data columns in metadata")
print(len(metadata_df.columns))

# Distribution of cancer types
if 'diagnostic' in df.columns:
    print("\nDiagnostic Distribution:")
    print(df['diagnostic'].value_counts())
    
    plt.figure(figsize=(12, 6))
    diagnostic_counts = df['diagnostic'].value_counts()
    diagnostic_counts.plot(kind='bar', color='coral')
    plt.xlabel('Diagnostic')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    figures_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "diagnostic_distribution.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {figure_path}")

# Cohen Kappa Score
df_pair = df[['hair_1', 'hair_2', 'hair_3', 'hair_4', 'hair_5']].dropna()

print("_______Kappa score of hair_______")
ck_12 = round(cohen_kappa_score(df_pair['hair_1'],df_pair['hair_2']),2)
ck_13 = round(cohen_kappa_score(df_pair['hair_1'],df_pair['hair_3']),2)
ck_14 = round(cohen_kappa_score(df_pair['hair_1'],df_pair['hair_4']),2)
ck_15 = round(cohen_kappa_score(df_pair['hair_1'],df_pair['hair_5']),2)
ck_23 = round(cohen_kappa_score(df_pair['hair_2'],df_pair['hair_3']),2)
ck_24 = round(cohen_kappa_score(df_pair['hair_2'],df_pair['hair_4']),2)
ck_25 = round(cohen_kappa_score(df_pair['hair_2'],df_pair['hair_5']),2)
ck_34 = round(cohen_kappa_score(df_pair['hair_3'],df_pair['hair_4']),2)
ck_35 = round(cohen_kappa_score(df_pair['hair_3'],df_pair['hair_5']),2)
ck_45 = round(cohen_kappa_score(df_pair['hair_4'],df_pair['hair_5']),2)
ck_sum = ck_12+ck_13+ck_14+ck_15+ck_23+ck_24+ck_25+ck_34+ck_35+ck_45
print(f" Mean: {ck_sum / 10}")

# Graph of annotations
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Hair annotations
hair_cols = [col for col in df_pair.columns if 'hair' in col]
hair_counts = df_pair[hair_cols].apply(pd.Series.value_counts).fillna(0).astype(int)
hair_counts.T.plot(kind='bar', ax=ax)
ax.set_xlabel('Annotator')
ax.set_ylabel('Count')
ax.legend(title='Hair Level')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
figures_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
figure_path = figures_dir / "Hair Annotations by Annotator"
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {figure_path}")

# Hair mean for cancerous and benign
cancer_labels = ['BCC', 'SCC', 'MEL']
df['cancer_flag'] = df['diagnostic'].apply(lambda diagnosis: "Cancerous" if diagnosis in cancer_labels else "Benign")
df['hair_mean'] = df[['hair_1', 'hair_2', 'hair_3', 'hair_4', 'hair_5']].mean(axis=1)
hair_by_cancer_flag = df.groupby('cancer_flag')['hair_mean'].mean()
print("\nHair mean by cancer classification:")
print(hair_by_cancer_flag)