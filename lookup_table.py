# Scannet classes
pred_classes = {
    0: "cabinet",
    1: "bed",
    2: "chair",
    3: "sofa",
    4: "table",
    5: "door",
    6: "window",
    7: "bookshelf",
    8: "picture",
    9: "counter",
    10: "desk",
    11: "curtain",
    12: "refrigerator",
    13: "showercurtain",
    14: "toilet",
    15: "sink",
    16: "bathtub",
    17: "garbage_bin"
}

# Ground truths in our dataset
gt_classes = {
    1: "chair",
    2: "bookshelf",
    3: "table"
    
}

#colors used for visualization of predictions
color_list = [[255,255,180], [255,0,0],[127,238,212], [0,255,255], [0,255,0], [100,100,0], [255,0,0],    # window las on this row
               [255,255,180], [255,0,0], [255,0,0], [128,0,128], [255,0,0], [255,0,0],[255,0,0], 
               [255,0,0],[255,0,0], [255,0,0],[255,0,0]]
