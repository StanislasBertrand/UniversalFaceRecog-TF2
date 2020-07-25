import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from icrawler.builtin import BingImageCrawler

def download_images(celebs_file, save_dir):
    celebrities_list = []
    with open(celebs_file) as f:
        for line in f:
            celebrities_list.append(line.strip())

    for celeb in celebrities_list:
        images_dir = os.path.join(save_dir, celeb.lower().replace(" ", "_"))
        if not os.path.isdir(images_dir):
            os.mkdir(images_dir)
            bing_crawler = BingImageCrawler(downloader_threads=4,storage={'root_dir': images_dir})
            bing_crawler.crawl(keyword=celeb, filters=None, offset=0, max_num=1000) 
