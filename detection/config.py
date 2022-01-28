NUM_CLASSES = 10
WEIGHT_DECAY = 1e-5
INPUT_SHAPE = (288, 288, 3)
BRANCH_0_SCALE = 1.2
BRANCH_1_SCALE = 1.1
BRANCH_2_SCALE = 1.05
LR_DECAY_LATE = 0.1
LR = 1e-3
EPS = 1e-7
WARMUP_EPOCHS = 10
EPOCHS = 200
BATCH_SIZE = 128
train_txt_path = "C:/Users/sangmin/Desktop/Dacon_LG/dataset/bdd/train.txt"
valid_txt_path = "C:/Users/sangmin/Desktop/Dacon_LG/dataset/bdd/valid.txt"
img_format = ".jpg"
save_dir = "./saved_models/detection/"
ANCHORS = []
with open("anchors.txt", 'r') as f:
    lines = f.readlines()
    line = lines[0].split(" ")
    for index, item in enumerate(line):
        wh = item.split(",")
        w = wh[0].replace("\n", "")
        h = wh[1].replace("\n", "")
        ANCHORS.append([float(w), float(h)])