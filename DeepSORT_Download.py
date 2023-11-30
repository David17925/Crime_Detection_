import gdown

# URL of the Google Drive file
file_url = "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf"

# Destination file path
output_path = "DeepSORT.zip"  # Change to the desired destination and file name

# Download the file
gdown.download(file_url, output_path, quiet=False)