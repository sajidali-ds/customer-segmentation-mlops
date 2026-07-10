import os
import subprocess
import zipfile
import shutil
import sys

DATASET_SLUG = "vijayuv/onlineretail"   # Kaggle dataset identifier
RAW_DIR = "data/raw"
TARGET_FILE = os.path.join(RAW_DIR, "online_retail.csv")
TMP_DIR = os.path.join(RAW_DIR, "_kaggle_tmp")


def check_kaggle_cli():
    try:
        import kaggle  # noqa: F401
        return True
    except (ImportError, OSError) as e:
        print("Kaggle package not ready:", e)
        return False


def download():
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"Downloading '{DATASET_SLUG}' from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", TMP_DIR],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("\nDownload failed. Common causes:")
        print("  - kaggle.json not placed correctly (see docstring above)")
        print("  - dataset slug changed on Kaggle's side; search 'Online Retail Dataset'")
        print("    on kaggle.com and update DATASET_SLUG in this script with the URL slug")
        sys.exit(1)

    # Unzip whatever came down
    for f in os.listdir(TMP_DIR):
        if f.endswith(".zip"):
            zip_path = os.path.join(TMP_DIR, f)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(TMP_DIR)

    # Find the CSV/XLSX inside and normalize it to data/raw/online_retail.csv
    candidate = None
    for f in os.listdir(TMP_DIR):
        if f.lower().endswith(".csv"):
            candidate = os.path.join(TMP_DIR, f)
            break
        if f.lower().endswith((".xlsx", ".xls")):
            # Some mirrors of this dataset ship as Excel -- convert to CSV
            import pandas as pd
            xlsx_path = os.path.join(TMP_DIR, f)
            df = pd.read_excel(xlsx_path)
            candidate = os.path.join(TMP_DIR, "converted.csv")
            df.to_csv(candidate, index=False)
            break

    if candidate is None:
        print("Could not find a CSV/XLSX in the downloaded dataset. Check", TMP_DIR)
        sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)
    shutil.copy(candidate, TARGET_FILE)
    shutil.rmtree(TMP_DIR)
    print(f"\nSaved real Kaggle dataset -> {TARGET_FILE}")
    print("Now run: python src/data_pipeline.py  then  python src/train.py")


if __name__ == "__main__":
    if not check_kaggle_cli():
        print("\nRun: pip install kaggle")
        print("Then set up kaggle.json as described in this file's docstring.")
        sys.exit(1)
    download()
