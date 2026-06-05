import requests
import tarfile
import os
import pandas as pd
import numpy as np

def download_file(url, output_path):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)
        print(f"{url} Downloaded successfully: {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def extract_tar_file(tar_path, extract_to):
    """Extract a .tar.gz file to a specific directory."""
    if tarfile.is_tarfile(tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
            print(f"Extracted to {extract_to}")
    else:
        print(f"{tar_path} is not a valid .tar.gz file")

# go to https://www.dropbox.com/scl/fo/wqxnbthlftupwe5xxgvq1/AFKlvHwmRbuhVqPMiowAMgQ?dl=0&e=2&rlkey=4sokuurl0xbcl4fhdnm5okvxl&st=00zo9bqp
# and save the HTML file to your local machine as a complete webpage HTML file
html_file = "/home/yorguin/projects/def-kjerbi/yorguin/datasets/MEG_LSD/352.html"


import re

# Load the HTML
with open(html_file, "r", encoding="utf-8") as f:
    html = f.read()

# Regex to find Dropbox hrefs that end with dl=0
pattern = r'href="(https://www\.dropbox\.com/scl/fo/[^"]*?dl=0)"'

# Extract matches
matches = re.findall(pattern, html)

# Print all matching links
for link in matches:
    print(link)

len(matches)

# Go from
"https://www.dropbox.com/scl/fo/wqxnbthlftupwe5xxgvq1/ANE-BwysbLIeEpFoowEzM0M/040914-2_LSD_20140904_Closed1.ds.tar.gz?rlkey=4sokuurl0xbcl4fhdnm5okvxl&amp;dl=0"
# to
"https://www.dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/ANE-BwysbLIeEpFoowEzM0M/040914-2_LSD_20140904_Closed1.ds.tar.gz?rlkey=4sokuurl0xbcl4fhdnm5okvxl&dl=0",


# get_filename = lambda x: os.path.basename(x.split('?rlkey')[0])
matches = [x.replace('https://www.dropbox.com/', 'https://www.dl.dropboxusercontent.com/') for x in matches]  # Replace 'dl=0' with 'dl=1' to get direct download links

to_download = [x for x in matches if True] # you could add a condition here if needed
get_filename = lambda x: os.path.basename(x.split('?rlkey')[0])
extract_dir = '/home/yorguin/scratch/data/MEG_LSDV2/meg_data'
temp_file =    '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/test.tar.gz'  # Local file name for downloaded file
# avoid using get_filename to set the download file in the loop if you are not interested in having the original tar.gz files

for i,dropbox_url in enumerate(to_download):
    # Create directory to extract files if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Download and extract the file
    this_file = get_filename(dropbox_url)
    print(f"Downloading {i+1}/{len(to_download)}: {this_file} ") #from {dropbox_url}

    if not os.path.exists(os.path.join(extract_dir, this_file.replace('.tar.gz', ''))):
        download_file(dropbox_url, temp_file)
        extract_tar_file(temp_file, extract_dir)
    else:
        print(f"File {this_file} already exists, skipping download.")
# you could add folder exists check so that you don't download the same file again
# 040914-2_LSD_20140904_Closed1.ds.tar.gz

#stat -c %G $HOME/scratch/data/MEG_LSDV2/*/

# this is from the link when you click on the download button that zips the dropbox folder
# https://www.dropbox.com/scl/fo/13lzzps647tuby74xvt3w/AGljSySi_VDb9HvHtqCIiLo?rlkey=saafhlfwhlo3hmodu6wv4ulvs&st=kbpxb4zo&dl=0
# (go there and download the zipped file), you should be able to get the dynamic link to download the zipped file
extra_url = 'https://uc3f9b10e2ea75e1db43342b0fab.dl.dropboxusercontent.com/zip_download_get/COAe5G6XDBYOd5r4vI0uOVGsOZ24a0cXWcjrMN-AVCPf1isKYWosjHPRHEuvVfTQ-1eo6J4C6csTaVnJBGP_q2wxG4Mtos3eXlNlssBzjXDGoQ?_download_id=621419945908794253496346351361550147085283638742638439996629616964&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1'

temp_file = '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/test.zip'
download_file(extra_url, temp_file)

import subprocess

subprocess.run(["unzip", "-o", temp_file, "-d", extract_dir], check=True)
# or try running it on the command line:
# unzip -o /home/yorguin/scratch/data/MEG_LSDV2/meg_data/test.zip -d /home/yorguin/scratch/data/MEG_LSDV2/meg_data