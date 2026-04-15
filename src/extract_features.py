import pandas as pd
from pathlib import Path
from skimage import io
import feature_A
import feature_B
import feature_C

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    def process_row(row):
        img_name = row['img_id']

        parts = img_name.split('_')
        id_value = f"{parts[0]}_{parts[1]}"

        masks_path = Path(__file__).resolve().parent.parent / "data" / "masks"
        img_path = Path(__file__).resolve().parent.parent / "data" / "imgs"
        mask_name = img_name[:-4] + "_mask" + img_name[-4:]
        mask_path = masks_path / mask_name

        image_path = img_path / img_name

        if not mask_path.exists():
            print(f"Missing image: {img_name}")
            return pd.Series([id_value, None, None, None])

        mask = io.imread(mask_path)
        img = io.imread(image_path)

        asymmetry = feature_A.get_asymmetry(mask)
        compactness = feature_B.get_compactness(mask)
        hue, saturation, brightness = feature_C.get_hsv_mean(img,mask)

        return pd.Series([id_value, asymmetry, compactness, hue, saturation, brightness])

    result = df.apply(process_row, axis=1)
    result.columns = ['ID', 'A_asymmetry', 'B_compactness', 'C_hue', 'C_saturation', 'C_brightness']
    result.to_csv(output_file, index=False)

process_csv(Path(__file__).resolve().parent.parent / "data" / "features.csv", 'output.csv')