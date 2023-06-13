import torch
#import pandas as pd
#from cnn import CNNNetwork
import torchaudio
from dataset import audioDataset
import psutil
#import matplotlib.pyplot as plt 
import torchvision

SAMPLE_RATE = 16000

# Input length. Changing this to higher length than original audio length will add padding to the tensor
NUM_SAMPLES = 32000

# Folder which contains the audio chunks 
AUDIO_DIR = "splitted_audio"

def mem():
    process = psutil.Process() #initiate only once
    memory_info = process.memory_info()
    rss = memory_info.rss
    rss_mb = rss / (1024 * 1024)
    print(f"Memory usage: {rss_mb} MB")

def predict(model, input, target, class_mapping):
    model.eval() 
    with torch.no_grad():
        predictions = model(input)
        #"predictions: ", predictions)
        predicted_index = predictions[0].argmax(0)
        #print("predicted_index: ", predicted_index)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def predict_softmax(model, input, target, class_mapping):
    model.eval() 
    with torch.no_grad():
        predictions = model(input)
        probabilities = torch.nn.functional.softmax(predictions[0]) # dim=0
        #print("pred ",predictions)
        #print("prob ", probabilities)
    return probabilities

def ex():
    cnn = torchvision.models.densenet121(num_classes = 6)
    #cnn_resnet50 = torchvision.models.resnet50(num_classes = 6)
    #cnn_alex = torchvision.models.alexnet(num_classes = 6)
    #cnn_mobilenetv3 = torchvision.models.mobilenet_v3_small(num_classes = 6)

    #print("cuda available?")
    #print(torch.cuda.is_available())
    if torch.cuda.is_available() == False:
        state_dict_dense = torch.load("models/dense121_emotion.pth", map_location=torch.device("cpu"))
        #state_dict_resnet = torch.load("models/resnet50_emotion.pth", map_location=torch.device("cpu"))
        #state_dict_mobilenet = torch.load("models/mobilenetv3_emotion.pth", map_location=torch.device("cpu"))

        #state_dict_mobilenet = torch.load("resnet_pre_emotion.pth", map_location=torch.device("cpu"))
    else:
        state_dict_dense = torch.load("models/dense121_emotion.pth")
        #state_dict_resnet = torch.load("models/resnet50_emotion.pth")
        #state_dict_mobilenet = torch.load("models/mobilenetv3_emotion.pth")

    #state_dict_alex = torch.load("alex_non_emotion.pth")

    cnn.load_state_dict(state_dict_dense)
    mem()

    #cnn_alex.load_state_dict(state_dict_alex)
    #cnn_resnet50.load_state_dict(state_dict_resnet)
    #cnn_mobilenetv3.load_state_dict(state_dict_mobilenet)

    cnn_emotion_list = [cnn]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    class_mapping = [
    "anger",
    "disdain",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    ]
    classes = "emotion"
    emotionData = audioDataset(
                    classes,
                    AUDIO_DIR,
                    mel_spectrogram,
                    NUM_SAMPLES,
                    SAMPLE_RATE,
                    "cpu")
    count = 0
    correct = 0

    predicted_list = []

    for x in range(1):
        predicted_list.append(
            {"anger": 0,
            "disdain":0,
            "fear":0,
            "happiness": 0,
            "neutral" :0,
            "sadness":0}
        )
    #print(predicted_list)
    y_true = []
    y_pred = [] 
    while count < len(emotionData):
        #print("cnn inf while loop")
        input = emotionData[count][0]
        target = emotionData[count][1]
        #print(input, target)
        input.unsqueeze_(0)
        #mem()

        for index, obj in enumerate(predicted_list):
            predicted = predict_softmax(cnn_emotion_list[index], input, target, class_mapping)
            
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

    #print("acc = ", correct / len(emotionData))
    #print("y_pred", y_pred)
    mem()
    return predicted_list


def ex_gender():
    mem()
    cnn = torchvision.models.densenet121(num_classes = 2)
    #cnn_resnet50 = torchvision.models.resnet50(num_classes = 2)
    #cnn_mobilenetv3 = torchvision.models.mobilenet_v3_small(num_classes = 2)
    #cnn_alex = torchvision.models.alexnet(num_classes = 2)

    if torch.cuda.is_available() == False:
        state_dict_dense = torch.load("models/dense121_gender.pth", map_location=torch.device("cpu"))
        #state_dict_resnet = torch.load("models/resnet50_gender.pth", map_location=torch.device("cpu"))
        #state_dict_mobilenet = torch.load("models/mobilenetv3_gender.pth", map_location=torch.device("cpu"))
    else:
        state_dict_dense = torch.load("models/dense121_gender.pth")
        #state_dict_resnet = torch.load("models/resnet50_gender.pth")
        #state_dict_mobilenet = torch.load("models/mobilenetv3_gender.pth")

    #state_dict_alex = torch.load("alex_pre_gender.pth")

    cnn.load_state_dict(state_dict_dense)
    #cnn_alex.load_state_dict(state_dict_alex)
    #cnn_resnet50.load_state_dict(state_dict_resnet)
    #cnn_mobilenetv3.load_state_dict(state_dict_mobilenet)
    cnn_gender_list = [cnn]

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

    for x in range(1):
        predicted_list_gender.append(
            {"female": 0,
            "male":0,
            }
        )

    classes = "gender"
    emotionData = audioDataset(
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
        input = emotionData[count][0]
        target = emotionData[count][1]
        input.unsqueeze_(0)
        for index, obj in enumerate(predicted_list_gender):
            #mem()
            predicted = predict_softmax(cnn_gender_list[index], input, target, class_mapping)
                    
            np_array = predicted.cpu().detach().numpy()

            obj["female"] = obj["female"] + np_array[0]
            obj["male"] = obj["male"] + np_array[1]
            obj["female"] = round(obj["female"],5)
            obj["male"] = round(obj["male"],5)

        count = count + 1
    mem()
    return predicted_list_gender


