import torch
#import pandas as pd
#from cnn import CNNNetwork
import torchaudio
from preprocess import preprocess
import psutil
#import matplotlib.pyplot as plt 
import torchvision

SAMPLE_RATE = 16000

# Input length. Changing this to higher length than original audio length will add padding to the tensor
NUM_SAMPLES = 32000

# Folder which contains the audio chunks 
AUDIO_DIR = "splitted_audio"

def predict_softmax(model, input, class_mapping):
    model.eval() 
    with torch.no_grad():
        predictions = model(input)
        probabilities = torch.nn.functional.softmax(predictions[0]) # dim=0
    return probabilities

def classify_emotions():
    cnn_densenet = torchvision.models.densenet121(num_classes = 6)
    cnn_resnet50 = torchvision.models.resnet50(num_classes = 6)
    cnn_alexnet = torchvision.models.alexnet(num_classes = 6)
    cnn_mobilenetv3 = torchvision.models.mobilenet_v3_small(num_classes = 6)

    if torch.cuda.is_available() == False:
        state_dict_densenet = torch.load("models/dense121_emotion.pth", map_location=torch.device("cpu"))
        state_dict_resnet = torch.load("models/resnet50_emotion.pth", map_location=torch.device("cpu"))
        state_dict_alexnet = torch.load("models/alexnet_emotion.pth", map_location=torch.device("cpu"))
        state_dict_mobilenet = torch.load("models/mobilenet_emotion.pth", map_location=torch.device("cpu"))
    else:
        state_dict_densenet = torch.load("models/dense121_emotion.pth")
        state_dict_resnet = torch.load("models/resnet50_emotion.pth")
        state_dict_alexnet = torch.load("models/alexnet_emotion.pth")
        state_dict_mobilenet = torch.load("models/mobilenetv3_emotion.pth")


    cnn_densenet.load_state_dict(state_dict_densenet)
    cnn_resnet50.load_state_dict(state_dict_resnet)
    cnn_alexnet.load_state_dict(state_dict_alexnet)
    cnn_mobilenetv3.load_state_dict(state_dict_mobilenet)

    cnn_emotion_list = [cnn_densenet, cnn_resnet50, cnn_alexnet, cnn_mobilenetv3]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    class_mapping = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    ]
    classes = "emotion"
    emotionData = preprocess(
                    classes,
                    AUDIO_DIR,
                    mel_spectrogram,
                    NUM_SAMPLES,
                    SAMPLE_RATE,
                    "cpu")
    count = 0
    correct = 0

    predicted_list = []
    
    for x in range(len(cnn_emotion_list)):
        predicted_list.append(
            {"anger": 0,
            "disdain":0,
            "fear":0,
            "happiness": 0,
            "neutral" :0,
            "sadness":0}
        )
    print(predicted_list)
    y_true = []
    y_pred = [] 
    while count < len(emotionData):
        input = emotionData[count]
        input.unsqueeze_(0)

        for index, obj in enumerate(predicted_list):
            predicted = predict_softmax(cnn_emotion_list[index], input, class_mapping)
            
            np_array = predicted.cpu().detach().numpy()

            obj["anger"] = obj["anger"] + np_array[0]
            obj["disdain"] = obj["disdain"] + np_array[1]
            obj["fear"] = obj["fear"] + np_array[2]
            obj["happiness"] = obj["happiness"] + np_array[3]
            obj["neutral"] = obj["neutral"] + np_array[4]
            obj["sadness"] = obj["sadness"] + np_array[5]

            obj["anger"] = round(obj["anger"],5)
            obj["disdain"] = round(obj["disdain"],5)
            obj["fear"] = round(obj["fear"],5)
            obj["happiness"] = round(obj["happiness"],5)
            obj["neutral"] = round(obj["neutral"],5)
            obj["sadness"] = round(obj["sadness"],5)

        count = count + 1
    return predicted_list


def classify_gender():
    cnn_densenet = torchvision.models.densenet121(num_classes = 2)
    cnn_resnet50 = torchvision.models.resnet50(num_classes = 2)
    cnn_alexnet = torchvision.models.alexnet(num_classes = 2)
    cnn_mobilenetv3 = torchvision.models.mobilenet_v3_small(num_classes = 2)

    if torch.cuda.is_available() == False:
        state_dict_densenet = torch.load("models/dense121_gender.pth", map_location=torch.device("cpu"))
        state_dict_resnet = torch.load("models/resnet50_gender.pth", map_location=torch.device("cpu"))
        state_dict_alexnet = torch.load("models/alexnet_gender.pth", map_location=torch.device("cpu"))
        state_dict_mobilenet = torch.load("models/mobilenet_gender.pth", map_location=torch.device("cpu"))
    else:
        state_dict_densenet = torch.load("models/dense121_gender.pth")
        state_dict_resnet = torch.load("models/resnet50_gender.pth")
        state_dict_alexnet = torch.load("models/alexnet_gender.pth")
        state_dict_mobilenet = torch.load("models/mobilenetv3_gender.pth")


    cnn_densenet.load_state_dict(state_dict_densenet)
    cnn_resnet50.load_state_dict(state_dict_resnet)
    cnn_alexnet.load_state_dict(state_dict_alexnet)
    cnn_mobilenetv3.load_state_dict(state_dict_mobilenet)   
    
    cnn_gender_list = [cnn_densenet, cnn_resnet50, cnn_alexnet, cnn_mobilenetv3]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    class_mapping = [
    "female",
    "male",
    ]

    predicted_list_gender = []

    for x in range(len(cnn_gender_list)):
        predicted_list_gender.append(
            {"female": 0,
            "male":0,
            }
        )

    classes = "gender"
    emotionData = preprocess(
                    classes,
                    AUDIO_DIR,
                    mel_spectrogram,
                    NUM_SAMPLES,
                    SAMPLE_RATE,
                    "cpu")

    count = 0
    correct = 0
    y_true = []
    y_pred = [] 
    while count < len(emotionData):
        input = emotionData[count]
        input.unsqueeze_(0)
        for index, obj in enumerate(predicted_list_gender):
            predicted = predict_softmax(cnn_gender_list[index], input, class_mapping)  
            np_array = predicted.cpu().detach().numpy()

            obj["female"] = obj["female"] + np_array[0]
            obj["male"] = obj["male"] + np_array[1]
            obj["female"] = round(obj["female"],5)
            obj["male"] = round(obj["male"],5)

        count = count + 1
    return predicted_list_gender


