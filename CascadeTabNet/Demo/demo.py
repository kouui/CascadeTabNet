from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# Load model
rootdir = "/nwork/kouui/works/cascadetabnet.20231118/CascadeTabNet/"
config_file = rootdir + 'Config/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
#checkpoint_file = rootdir + 'models/epoch_36.pth'
checkpoint_file = rootdir + 'models/epoch_36.v2.pth'
#checkpoint_file = rootdir + 'models/epoch_24.pth'
#checkpoint_file = rootdir + 'models/epoch_24.v2.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('Bordered', 'cell', 'Borderless') + model.CLASSES[3:]
# Test a single image 
#img = rootdir + "Demo/demo.png"
#img = rootdir + "data/rpa-ocr/60_4.jpg"
# Run Inference
#result = inference_detector(model, img)
#show_result_pyplot(img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)
#show_result_pyplot(model, img, result, score_thr=0.3,out_file=rootdir + "output/rpa-ocr/60_4.jpg")

from glob import glob
import os

foldername = "rpa-ocr"
folder = rootdir + f"data/{foldername}/"
outfolder = rootdir + f"output/{foldername}/"
imfiles = glob(folder+"*")
for img in imfiles:
    print(f"processing : {img}")
    try:
        result = inference_detector(model, img)
    except AttributeError:
        continue
    show_result_pyplot(model, img, result, score_thr=0.3,out_file=f"{outfolder}/{os.path.basename(img)}")
