# Web application for testing audio classification models

- This application combines React frontend, Flask backend and Torch machine learning.

- Record or upload your own audio files.

- Convolutional neural networks will process and classify your files. You will a receive visual representation of the results.

- Machine learning happens completely on server side, which makes this application's frontend very light to use.

<span style="color:yellow">Live version: https://cnn-audio.herokuapp.com
</span>

- Live version includes only 2 of the 4 CNN models and cpu-only version of torch due to memory and storage limitations.

# Prerequisites

- Node.js
- npm
- Python version (3.11.4 or higher)

# Getting Started

1. clone the repo

```sh
git clone https://github.com/arttuhaverinen/audio-classification-web-app.git
```

2. Install Python packages

```sh
pip install -r requirements.txt
```

3. Install NPM packages

```sh
cd client
npm install
```

4. Run application

- Frontend

```sh
cd client
npm start
```

- Backend
  - Run app.py

5. Creating production build (if needed)

```sh
cd client
npm run build
```

# Usage

- User can upload audio in the following ways:

  - Record your voice with your microphone
  - Upload wav or mp3 files from your device
  - Paste a youtube url

- Upload audio to the backend where it will be processed and analyzed

  - There is no limit in how long the audio files can be, but processing files longer than 1 minute might take some time.

- Displays user's classification results using charts

# Possible improvements

- Multiple file upload
- Download CSV-file from your results
- Customize which CNNs to use
- Increase UI customization

## Languages & Libraries

Languages

- JavaScript
- Python

Libraries:

- React
- Flask
- Torch, TorchAudio, TorchVision

## Additional information

The following code was used to train the models:
https://github.com/arttuhaverinen/torch-cnn-audio-classification

## Contact

haverinen994@gmail.com
