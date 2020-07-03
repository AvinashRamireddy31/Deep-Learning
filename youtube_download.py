import os
from pytube import YouTube

def downloadYouTube(videourl, path, filename):

    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive= False, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(path, filename)
     

videourl = 'https://www.youtube.com/watch?v=_gPcYovP7wc&list=PL_z_8CaSLPWekqhdCPmFohncHwz8TY2Go&index=7'
 
downloadYouTube(videourl, path='./videos/Dynamic_Programming', filename='7 Subset Sum Problem')
