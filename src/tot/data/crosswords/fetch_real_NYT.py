from nyt_crossword_downloader import RangeDownloader
from datetime import datetime

r = RangeDownloader(
    destination="~/puzzles",
    cookie_file="~/cookies.txt",
    date_folders=True,
    secs_btwn_queries=10,
)
r.download_date_range(datetime(2020,1,1), datetime(2020,1,31))
