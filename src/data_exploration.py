import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

metadata_csv = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"
df = pd.read_csv(metadata_csv)

if 'diagnostic' in df.columns:
    print("\nDiagnostic Distribution:")
    print(df['diagnostic'].value_counts())
    
    plt.figure(figsize=(12, 6))
    diagnostic_counts = df['diagnostic'].value_counts()
    diagnostic_counts.plot(kind='bar', color='coral')
    plt.title('Distribution of Cancer Types (Diagnostic)', fontsize=14, fontweight='bold')
    plt.xlabel('Diagnostic')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    figures_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "diagnostic_distribution.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {figure_path}")