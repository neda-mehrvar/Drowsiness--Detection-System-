import sys
import threading
import numpy as np

import cv2
import torch
import torch.nn.functional as F
from matplotlib.image import interpolations_names
from nbformat.v1.nbjson import JSONReader
from  torchvision import transforms
from pathlib import Path
import mediapipe as mp
import os

from config import Config
from realtime_config import RealtimeConfig
from models_loss import build_model
from smooter import ProbSmoother


class Alarm:
    def __init__(self,freq=2000,dur_ms=800):
        self.freq=freq
        self.dur_ms=dur_ms

        def _beep_win(self):
            try:
                import winsound
                winsound.Beep(self.freq, self.dur_ms)
            except Exception:
                print(">>>>>ALARM !!!! Drowsiness detected.")
        def _beep_other(self):
            print(">>>> ALARM !!!! Drowsiness detected.")
        def trigger(self):
            def _runner():
                if sys.platform.startswith("win"):
                    self._beep_win()
                else:
                    self._beep_other()
            threading.Thread(target=_runner, daemon=True)



class  LeftEyeLandmarkDetector:

        def __init__(self,rt_jf:RealtimeConfig):
            self.rt_jf=rt_jf
            self.eye_margin = self.rt_jf["eye_margin"]
            self.mp_face=mp.solutions.face_mesh
            self.mesh=self.mp_face.FaceMesh(static_image_mode=False,max_num_faces=1,
                                            refine_landmarks=True,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
        def _crop_bbox_from_landmarks(self,frame_bgr,landmarks):
            h,w=frame_bgr.shape[:2]
            pts=np.array([(int(l.x*w),int(l.y*h)) for l in landmarks ], dtype=np.int32)
            x1,y1=pts[:,0].min(), pts[:,1].min()
            x2,y2=pts[:,0].max(), pts[:,1].max()

            dw=int((x2-x1)*self.eye_margin)
            dh=int((y2-y1)*self.eye_margin)
            x1=max(0,x1-dw)
            y1=max(0,y1-int(dh*3.5))#y1=max(0,y1-dh)
            x2 = min(w, x2 + dw)
            y2 = min(h, y2 + int(dh*3.5))#y2 = min(h, y2 + dh)

            if x2<=x1 or y2<=y1:
                return None, None
            eye=frame_bgr[y1:y2,x1:x2].copy()
            eye=self.enhance_eye(eye,out_size=(60,36))
            return eye,(x1,y1,x2-x1,y2-y1)

        def enhance_eye(self,eye_bgr,out_size,use_clahe=True,alpha=None,beta=None,gauss_kernel=None,gauss_sigma=None):

            if eye_bgr is None or eye_bgr.size==0:
                return None
            eye = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)

            if use_clahe:
                clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                eye=clahe.apply(eye)
            else:
                eye=cv2.equalizeHist(eye)
            eye=cv2.convertScaleAbs(eye,alpha=self.rt_jf["alpha"],beta=self.rt_jf["beta"])
            gauss_kernel=self.rt_jf["gauss_kernel"]
            gauss_sigma = self.rt_jf["gauss_sigma"]
            if gauss_kernel and gauss_kernel % 2 == 1:
                eye=cv2.GaussianBlur(eye,(gauss_kernel,gauss_kernel),gauss_sigma)
            eye=cv2.resize(eye,out_size,interpolation=cv2.INTER_AREA)
            return eye

        def detect_left_eye(self,frame_bgr):
            rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
            res=self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None,None
            face=res.multi_face_landmarks[0]
            left_eye_lms=[face.landmark[i] for i in self.rt_jf["right_eye_idx"]]
            eye_img, box=self._crop_bbox_from_landmarks(frame_bgr,left_eye_lms)
            return eye_img, box

        def cap(self):
            cap = cv2.VideoCapture(0)
            detector =LeftEyeLandmarkDetector(self.rt_jf)
            saved = 0
            max_saved = 100
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                eye_img, box = detector.detect_left_eye(frame)

                if eye_img is not None:
                    cv2.imshow("left eye", eye_img)
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    if saved < max_saved:
                        filename =os.path.join(self.rt_jf["frames_folder"] ,f"frame_{saved + 1:02d}.png")
                        cv2.imwrite(filename, eye_img)
                        print(f" saved  done : {filename}")
                        saved += 1
                    if saved >= max_saved:
                        print(" 5"
                              " frames are saved.")
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()


def main():
    rt_jf=RealtimeConfig("realtime_config.json")
    detect_eye=LeftEyeLandmarkDetector(rt_jf)
    detect_eye.cap()
if __name__=="__main__":
    main()


















'''
class FahrerMuedigkeitSystem:
    def __init__(self, model, device,rt_jf: RealtimeConfig, jf:Config):

        self.rt_jf = rt_jf
        self.jf = jf

        checkpoint_path = self.jf["best_path"]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state=checkpoint.get("model_state",checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

        self.eye_side=str(self.rt_jf["eye_side"]).lower()
        self.mirror=bool(self.rt_jf["mirror"])
        self.tf=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(self.jf["img_size"]),int(self.jf["img_size"]))),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        self.eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye_tree_eyeglasses.xml")
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Haar cascades not found. Check OpenCv installation")

        self.smoother=ProbSmoother(window_sec=float(rt_jf["window_sec"]), thresh=float(rt_jf["thresh"]), main_alert_sec=float(rt_jf["main_alert_sec"]))

        def _pick_eye(self,eyes,prefer_left=True):
            #eyes:list of (ex,ey,ew,eh)
            if not eyes:
                return None
            eyes_sorted=sorted(eyes,key=lambda  e: e[0])
            return eyes_sorted[0] if prefer_left else eyes_sorted[-1]
        def _detect_left_or_not_eye_roi(self,frame_bgr):
            if self.mirror:
                frame_bgr=cv2.flip(frame_bgr,1)
            gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
            faces=self.face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(80,80))
            if len(faces)==0:
                return None
            faces=sorted(faces,key=lambda  f: f[2]*f[3], reverse=True)
            (x,y,w,h)=faces[0]
            face_gray=gray[y:y+h,x:x+w]
            face_bgr=frame_bgr[y:y+h,x:x+w]
            eyes=sele.eye_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(20,20))
            if len(eyes)==0:
                return None
            prefer_left=(self.eye_side=="left")
            chosen=self._pick_eye(list(eyes),prefer_left=prefer_left)
            if chosen is None:
                return None
            (ex,ey,ew,eh)=chosen
            roi=face_bgr[ey:ey+eh,ex:ex+ew]
            return roi if roi.size > 0 else None

        @torch.no_grad()
        def predict
'''




















