from flask import Flask
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask import request
from flask.helpers import send_from_directory
import os
#import io
import soundfile as sf
#import base64
from werkzeug.utils import secure_filename
#from moviepy.editor import *
from pydub import AudioSegment
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub.utils import make_chunks
#import dataset
import cnn_inference
import glob
from pytube import YouTube
from pathlib import Path
#import importlib.util
#import sys
import psutil
#import torch
import time

# FUNCTIONS

#app = Flask(__name__)
app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

app.config['DEBUG'] = True

@app.route("/mem", methods=["GET"])
def mem():
    process = psutil.Process() #initiate only once
    memory_info = process.memory_info()
    rss = memory_info.rss
    rss_mb = rss / (1024 * 1024)
    print(f"Memory usage: {rss_mb} MB")
    return jsonify(f"Memory usage: {rss_mb} MB")

#def getmem():
#    process = psutil.Process() #initiate only once
#    memory_info = process.memory_info()
#    rss = memory_info.rss
#    rss_mb = rss / (1024 * 1024)
#    print(f"Memory usage: {rss_mb} MB")
#    #return jsonify(f"Memory usage: {rss_mb} MB")

import threading

#def printit():
#  threading.Timer(5.0, printit).start()
#  getmem()

#printit()


# For production
@app.route("/")
@cross_origin()
def serve():
  #return send_from_directory(app.static_folder, "index.html")
  return app.send_static_file("index.html")
  

# For recorded audio
@app.route("/blob", methods=["POST"])
@cross_origin()
def blob ():

  print(request.files)
  print("file: ", request.files["audiofile"])

  f = request.files["audiofile"]
  f.save(os.path.join("saved_audio", secure_filename("file.wav")))

  myaudio = AudioSegment.from_file(os.path.join("saved_audio","file.wav")) 
  split_wav_file(myaudio, 2000)
  values = use_model()

  data = {"values": values}
  print("data", data)
  return jsonify(data)
  return []

# Youtube urls
@app.route("/youtube", methods=["POST"])
@cross_origin()
def youtube():
  print("youtube")

  print("request: ", request.data)
  yt_link = request.data.decode("UTF-8")
  url = YouTube(yt_link)
  print("downloading....")
  video = url.streams.filter(only_audio=True).first()
  cwd = Path.cwd()
  path_to_download_folder = cwd / "saved_audio"
  print(path_to_download_folder)
  video.download(path_to_download_folder, filename="file.mp4")

  sound = AudioSegment.from_file(os.path.join("saved_audio","file.mp4") , "mp4") 
  sound.export(os.path.join("saved_audio", "file.wav"), "wav")

  myaudio = AudioSegment.from_file(os.path.join("saved_audio","file.wav") , "wav") 
  split_wav_file(myaudio, 2000)

  values = use_model()

  delete_files()

  data = {"values": values}
  print("data", data)
  return jsonify(data)

@app.route("/test_get", methods=["GET"])
@cross_origin()
def test_get(): 
  return jsonify("test get")

@app.route("/test_post", methods=["POST"])
@cross_origin()
def test_post(): 
  return jsonify("test post")

# Regular mp3/wav files
@app.route("/wav", methods=["POST"])
@cross_origin()
def wav():

  blob = request.data
  print("wav post")
  print(request.files)

  f = request.files["file"]
  filetype = f.filename.split('.')[-1]
  print(f.filename)
  print(filetype)

  if filetype == "mp3":
    print("mp3 file type")
    f.save(os.path.join("saved_audio", secure_filename("file.mp3")))
    sound = AudioSegment.from_file(os.path.join("saved_audio","file.mp3") , "mp3") 
    sound.export(os.path.join("saved_audio", "file.wav"), "wav")

  if filetype == "wav":
    f.save(os.path.join("saved_audio", secure_filename("file.wav")))
    print(os.listdir())


  myaudio = AudioSegment.from_file(os.path.join("saved_audio","file.wav") , "wav") 
  split_wav_file(myaudio, 2000)

  values = use_model()

  print("values ", values)
  
  delete_files()

  data = {"values": values}
  print("data", data)
  return jsonify(data)


# FUNCTIONS

# Split audio to chunks, chunk_len determines the length
def split_wav_file(myaudio, chunk_len):

  chunk_length_ms = chunk_len 
  chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

  for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    #print ("exporting", chunk_name)
    chunk.export(os.path.join("splitted_audio", chunk_name), format="wav")

# Delete user's uploaded files
def delete_files():
  files = glob.glob('splitted_audio/*')
  for f in files:
    os.remove(f)
  files = glob.glob('saved_audio/*')
  for f in files:
    os.remove(f)
  print("removed")
  
  files = glob.glob('chunks/*')
  for f in files:
    os.remove(f)
  print("removed")
  print("removed")


# uses the created model
def use_model():
  #getmem()
  emotion_values = cnn_inference.ex()
  gender_values = cnn_inference.ex_gender()
  #getmem()
  print("gender_values, ", gender_values)
  values = [emotion_values, gender_values]
  mem()
  return values

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, threaded=False)
