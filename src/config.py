import torch

BATCH_SIZE = 4
RESIZE_TO = 512
NUM_EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = "../data/train"
VALID_DIR = "../data/valid"
TEST_DIR = "../data/test"
OUT_DIR = "../outputs"

SAVE_PLOTS_EPOCH = 10 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 10 # save model after these many epochs

CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5

VISUAL_TRANSFORMED_IMAGES = False

