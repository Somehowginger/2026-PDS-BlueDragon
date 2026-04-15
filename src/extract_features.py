import pandas as pd
from pathlib import Path
from feature_A import get_assymetry
from feature_B import get_compactness
#from feature_C import get_hsv_means

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    masks_path = Path(__file__).resolve().parent.parent / "data" / "masks"

    def process_row(row):
        img_name = row['img_id']

        parts = img_name.split('_')
        id_value = f"{parts[0]}_{parts[1]}"

        image_path = masks_path / img_name

        if not image_path.exists():
            print(f"Missing image: {img_name}")
            return pd.Series([id_value, None, None, None])

        assymetry = get_assymetry(image_path)
        compactness = get_compactness(image_path)
        #colour = get_hsv_means()

        return pd.Series([id_value, assymetry, compactness, None])
    
    df[['ID', 'feature_A', 'feature_B', 'feature_C']] = df.apply(process_row, axis=1)

    df[['ID', 'feature_A', 'feature_B', 'feature_C', ]].to_csv(output_file, index=False)

process_csv(Path(__file__).resolve().parent.parent / "data" / "features.csv", 'output.csv')