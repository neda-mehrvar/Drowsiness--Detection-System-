import os.path

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import glob

from models_loss import build_model
from config import Config
from realtime_config import RealtimeConfig



class FrameToModel:
    def __init__(self, rt_jf: RealtimeConfig, jf:Config):
        self.rt_jf = rt_jf
        self.jf=jf
        self.tf=transforms.Compose([
                transforms.Resize((self.jf["img_size"],self.jf["img_size"])),
                transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def preprocess_image(self,path_img):
        img=cv2.imread(path_img)
        if img is None:
            print(f"Could not read {path_img}")
            return None
        if img.ndim==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil=Image.fromarray(img)
        return self.tf(pil)

    def images_to_tensor(self):
        folder=self.rt_jf["frames_folder"]
        self.paths=sorted(glob.glob(fr"{folder}\frame_*.png"))#List of image_paths
        print("Sample paths_images:",self.paths[:2])
        tensors=[]
        for p in self.paths:
            t=self.preprocess_image(p)
            if t is not None:
               tensors.append(t)
        if not tensors:
            raise  RuntimeError("No valid images found!")
        self.batch=torch.stack(tensors,dim=0).to(self.device)
        print("Batch shape:",self.batch.shape)
        return self.batch

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.jf).to(self.device)
        checkpoint_path = self.jf["best_path"]
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        return self.model
    @torch.no_grad()
    def predict_batch(self,label_map=None):
        if label_map is None:
            label_map={1:"closed", 0:"open"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.batch=self.batch.to(self.device)
        logits=self.model(self.batch)
        probs=torch.softmax(logits,dim=1)
        preds=probs.argmax(dim=1).cpu().numpy()
        probs=probs.cpu().numpy()
        self.labels=[label_map[int(p)] for p in preds]
        final_pred=int(max(set(preds),key=list(preds).count))
        final_label=label_map[final_pred]
        print("frame-labels:",self.labels[:len(self.paths)],"...")
        print("final_majority:",final_label)
        return preds,probs, self.labels,final_label

    def save_frames_with_labels(self):
        out_folder=self.rt_jf["out_folder"]
        for i, (path,label) in enumerate(zip(self.paths,self.labels), start=1):
            img=cv2.imread(path)
            if img is None:
                continue
            h,w=img.shape[:2]
            bar_height=50
            bar=np.ones((bar_height,w,3),dtype=np.uint8)*255

            font=cv2.FONT_HERSHEY_SIMPLEX
            scale=0.5
            color=(0,255,0) if label=="open" else (0,0,255)
            thickness=1


            text_size,_=cv2.getTextSize(label,font,scale,thickness)
            text_x=(w-text_size[0])//2
            text_y=(bar_height+text_size[1])//2

            cv2.putText(bar,label,(text_x,text_y),font,scale,color,thickness,cv2.LINE_AA)
            labeled_img=np.vstack([bar,img])
            out_path=os.path.join(out_folder,f"labeled_{i:03d}.png")
            cv2.imwrite(out_path,labeled_img)
        print(f"{len(self.labels)}  -----    {out_folder}")

def main():
    rt_jf=RealtimeConfig("realtime_config.json")
    jf=Config("config.json")

    f2m=FrameToModel(rt_jf,jf)
    f2m.load_model()
    batch=f2m.images_to_tensor()
    f2m.predict_batch()
    f2m.save_frames_with_labels()

if __name__=="__main__":
    main()

