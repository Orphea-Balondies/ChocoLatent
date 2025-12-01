import kagglehub

# Download latest version
path = kagglehub.dataset_download("marcinrutecki/old-photos")

print("Path to dataset files:", path)