import os
from PIL import Image

base_dir = r"D:\FogMap\SMOG4000"
corrupted_files = []

print("Scanning dataset for corrupted images...")

# Walk through all folders and subfolders
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(root, file)
            try:
                # Try to open and verify the image
                img = Image.open(file_path)
                img.verify() 
            except Exception as e:
                print(f"Corrupted file detected: {file_path}")
                corrupted_files.append(file_path)

print(f"\nScan complete. Found {len(corrupted_files)} corrupted files.")

# Automatically delete the corrupted files
if len(corrupted_files) > 0:
    for f in corrupted_files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")
    print("All corrupted files removed. You can now run train.py!")
else:
    print("No corrupted files found.")