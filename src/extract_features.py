import pandas as pd
from pathlib import Path
from skimage import io
import feature_A
import feature_B

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    def process_row(row):
        img_name = row['img_id']

        parts = img_name.split('_')
        id_value = f"{parts[0]}_{parts[1]}"

        masks_path = Path(__file__).resolve().parent.parent / "data" / "masks"
        mask_name = img_name[:-4] + "_mask" + img_name[-4:]
        mask_path = masks_path / mask_name

        if not mask_path.exists():
            print(f"Missing image: {img_name}")
            return pd.Series([id_value, None, None, None])

        mask = io.imread(mask_path)
        compactness = feature_B.get_compactness(mask)

        return pd.Series([id_value, None, compactness, None])

    result = df.apply(process_row, axis=1)
    result.columns = ['ID', 'feature_A', 'feature_B', 'feature_C']
    result.to_csv(output_file, index=False)

process_csv(Path(__file__).resolve().parent.parent / "data" / "features.csv", 'output.csv')