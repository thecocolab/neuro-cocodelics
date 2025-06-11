from pathlib import Path

import gdown

# The folder ID from your share link
folder_url = "https://drive.google.com/drive/folders/1SNG7EK422o7qok2kQtUOJAetW3c9pNCK"

# Download all files in the folder into the current directory (recursive)
gdown.download_folder(folder_url, output=(Path(__file__).parent.parent / "local_data").as_posix(), use_cookies=False)
