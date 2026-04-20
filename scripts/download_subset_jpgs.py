# scripts/download_subset_jpgs.py

from pathlib import Path
import urllib.request
import urllib.error
import getpass
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JPG_LIST_PATH = PROJECT_ROOT / "data" / "processed" / "subset_jpg_paths.txt"
DOWNLOAD_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg_subset"

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


def build_auth_opener(username: str, password: str):
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(
        realm=None,
        uri="https://physionet.org/",
        user=username,
        passwd=password
    )
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)


def download_one_file(rel_path: str, max_retries: int = 3):
    rel_path = rel_path.strip().replace("\\", "/")
    url = BASE_URL + rel_path
    out_path = DOWNLOAD_ROOT / Path(rel_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return "skipped_existing", rel_path

    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, out_path)
            return "downloaded", rel_path
        except urllib.error.HTTPError as e:
            return f"http_error_{e.code}", rel_path
        except Exception as e:
            if attempt == max_retries:
                return f"failed_{type(e).__name__}", rel_path
            time.sleep(2)

    return "failed_unknown", rel_path


def main():
    print("=== DOWNLOAD SUBSET JPG IMAGES ===")
    print(f"Reading path list from: {JPG_LIST_PATH}")
    print(f"Download root: {DOWNLOAD_ROOT}")

    if not JPG_LIST_PATH.exists():
        raise FileNotFoundError(f"Missing JPG path list: {JPG_LIST_PATH}")

    with open(JPG_LIST_PATH, "r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    print(f"Total paths to process: {len(rel_paths)}")

    username = input("Enter your PhysioNet username: ").strip()
    password = getpass.getpass("Enter your PhysioNet password (input hidden): ")

    build_auth_opener(username, password)

    downloaded = 0
    skipped = 0
    failed = 0
    failure_examples = []

    for idx, rel_path in enumerate(rel_paths, start=1):
        status, rp = download_one_file(rel_path)

        if status == "downloaded":
            downloaded += 1
        elif status == "skipped_existing":
            skipped += 1
        else:
            failed += 1
            if len(failure_examples) < 20:
                failure_examples.append((rp, status))

        if idx % 100 == 0 or idx == len(rel_paths):
            print(
                f"[{idx}/{len(rel_paths)}] "
                f"downloaded={downloaded}, skipped={skipped}, failed={failed}"
            )

    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"Downloaded new files : {downloaded}")
    print(f"Skipped existing     : {skipped}")
    print(f"Failed               : {failed}")

    if failure_examples:
        print("\nSample failures:")
        for rp, status in failure_examples:
            print(f"  - {rp} -> {status}")

    print("\nDownload step completed.")


if __name__ == "__main__":
    main()