import shutil
from pathlib import Path

TARGET_DIRS = [
    "encoded",
    "outputs/clips"
]

def clear_directory(path):
    path = Path(path)
    if not path.exists():
        print(f"{path} does not exist. Skipping.")
        return
    print(f"Clearing {path}...")
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"{path} cleared.")

def main():
    for d in TARGET_DIRS:
        clear_directory(d)
    print("\nData reset complete.")

if __name__ == "__main__":
    main()