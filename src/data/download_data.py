import subprocess
import os

from prepare_data import split_data


def download_file_with_wget(url, destination_path):
    try:
        # Using subprocess to run the wget command
        file_name = os.path.split(url)[-1]
        destination = os.path.join(destination_path, file_name)
        subprocess.run(['wget', url, '-O', destination], check=True)
        print(f"File downloaded successfully to: {destination}")
        return destination
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        raise
    

def unzip_file_with_bzip2(file_path):
    try:
        # Using subprocess to run the bzip2 command
        subprocess.run(['bzip2', '-d', file_path], check=True)
        out_file_path = ".".join(file_path.split(".")[:-1])
        print(f"File unziped successfully to: {out_file_path}")
        return out_file_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        raise
    

urls = {
    "court": "https://lang.org.ua/static/downloads/ubertext2.0/court/cleansed/ubertext.court.filter_rus_gcld+short.text_only.txt.bz2",
    "fiction": "https://lang.org.ua/static/downloads/ubertext2.0/fiction/cleansed/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2",
    "news": "https://lang.org.ua/static/downloads/ubertext2.0/news/cleansed/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2",
    "social": "https://lang.org.ua/static/downloads/ubertext2.0/social/cleansed/ubertext.social.filter_rus_gcld+short.text_only.txt.bz2",
    "wiki": "https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/cleansed/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2"
}

# split_ratios = {
#     "court": 0.999,
#     "fiction": 0.99,
#     "news": 0.9999,
#     "social": 0.999,
#     "wiki": 0.9995
# }

train_size = 1000
test_size = 50

for data_name in urls:
    # split_ratio = split_ratios[data_name]
    url = urls[data_name]
    file_name = os.path.split(url)[-1]
    file_path = os.path.join("data", file_name)
    file_path = ".".join(file_path.split(".")[:-1])
    if os.path.isfile(file_path):
        split_data(file_path, "splitted_data", data_name, min_len=8192, train_size=train_size, test_size=test_size)
    else:
        file_path = download_file_with_wget(url, "data")
        file_path = unzip_file_with_bzip2(file_path)
        split_data(file_path, "splitted_data", data_name, min_len=8192, train_size=train_size, test_size=test_size)
