# Model parameters
NUM_CLASSES = 8 + 1
MAX_SIZE = 550
# MODEL_PATH = './model/weights/result_101_anchor3_sel2/model.pth'
## backbone
BACKBONE_RESNET_101 = 0
BACKBONE_RESNET_50 = 1
LAYERS = {
    BACKBONE_RESNET_101: [3, 4, 23, 3],
    BACKBONE_RESNET_50: [3, 4, 6, 3]
}
backbone_type = BACKBONE_RESNET_101
## prediction select (default: [1, 2, 3])
PREDICTION_SELECTED_LAYERS = [1, 2, 3]
NUM_DOWNSAMPLE = 2
## anchor ratio (default: [1, 0.5, 2])
ANCHOR_ASPECT_RATIO = [1, 0.5, 2]

# Train parameters
PRETRAINED_MODEL_PATHS = {
    BACKBONE_RESNET_101: 'model/weights/resnet101_reducedfc.pth',
    BACKBONE_RESNET_50: 'model/weights/resnet50-19c8e357.pth'
}

DISCARD_BOX_WIDTH = 4 / MAX_SIZE
DISCARD_BOX_HEIGHT = 4 / MAX_SIZE

# Augmentation parameters
AUG_PHOTOMETRIC_DISTORT = True
AUG_EXPAND = True
AUG_RANDOM_SAMPLE_CROP = True
AUG_RANDOM_MIRROR = True
AUG_RANDOM_FLIP = False
AUG_PRESERVE_ASPECT_RATIO = False
AUG_MASK_SIZE = 16
AUG_USE_GT_BBOXES = False


# Test parameters
TOP_K = 15
CONF_THRESH = 0.05
NMS_THRESH = 0.5
SCORE_THRESHOLD = 0.15

# Visualize parameters
MASK_ALPHA = 0.45
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

# Input image parameters
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

