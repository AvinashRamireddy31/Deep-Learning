import os
from pytube import YouTube

def downloadYouTube(videourl, path, filename):

    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive= True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(path)
    yt.set_filename(filename)

videourl = 'https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6'
 
downloadYouTube(videourl, path='./video/PyTorch Tutorial', filename=' 06 - Training Pipeline: Model, Loss, and Optimizer')
