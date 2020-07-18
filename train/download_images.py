import os
from icrawler.builtin import BingImageCrawler
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('celebs_file', './data/celebrities.txt', 'file containing list of celebrities')
flags.DEFINE_string('save_dir', './data/images/bing_images/', 'output dir to save crawled images in')


def _main(_argv):
    celebrities_list = []
    with open(FLAGS.celebs_file) as f:
        for line in f:
            celebrities_list.append(line.strip())

    for celeb in celebrities_list:
        images_dir = os.path.join(FLAGS.save_dir, celeb.lower().replace(" ", "_"))
        if not os.path.isdir(images_dir):
            os.mkdir(images_dir)
            bing_crawler = BingImageCrawler(downloader_threads=4,storage={'root_dir': images_dir})
            bing_crawler.crawl(keyword=celeb, filters=None, offset=0, max_num=1000) 

if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
