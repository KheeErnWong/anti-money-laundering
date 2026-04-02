import subprocess
import zipfile

from src.config import RAW_DIR


def download_data():
    """Download IBM AML HI-Small dataset from Kaggle.

    Returns:
        Path to the downloaded CSV file.
    """
    target = RAW_DIR / "HI-Small_Trans.csv"

    if target.exists():
        print("Dataset has already been downloaded")
        return target

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
            "-f",
            "HI-Small_Trans.csv",
            "-p",
            str(RAW_DIR),
        ],
        check=True,
    )

    for zf in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    print(f"Dataset downloaded to {target}")

    return target


if __name__ == "__main__":
    download_data()
