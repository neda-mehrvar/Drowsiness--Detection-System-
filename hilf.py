





from pathlib import Path
from realtime_config import RealtimeConfig

root=Path("C:\\Users\\schulung\\Desktop\\P_MFT")
rt=RealtimeConfig(root/"cods\\realtime_config.json")
rt.show()
print("#####",rt["eye_side"])
#import pandas as pd
#from pathlib import Path
#root=Path(r"C:\Users\schulung\Desktop\P_MFT\Data")
#train_csv=root/"train.csv"
#df=pd.read_csv(train_csv)

#print("len_train_csv",len(df))



#test Dataset
'''print("len_train_dataset",len(train_dataset))
img,label=train_dataset[0]
print("img",img.shape)
print("label",label)

imgs,labels=next(iter(torch.utils.data.DataLoader(train_dataset,batch_size=8)))
grid=torchvision.utils.make_grid(imgs,nrow=4,normalize=True)

plt.figure(figsize=(8,8))
plt.imshow(grid.permute((1,2,0)))
plt.title("images with labels"+str(labels.tolist()))
plt.axis("off")
plt.show()'''


#test Dataloader
'''imges,labels=next(iter(train_loader))
print("Batch images:",imges.shape)
print("Batch labels shape:",labels.shape)
print("first 10 labels:",labels[:10])
'''
###
'''
#sapmler=sampler   <---> shuffle=True  (entweder-oder)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights=weights.to(device)
labels_tensor=torch.tensor(train_dataset.labels,dtype=torch.long).to(device)
sample_weights=torch.where(labels_tensor==0, weights[0],weights[1]).double()
sampler=WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)
'''


'''def compute_class_weights_and_loss(train_csv,device):
    counts = pd.read_csv(train_csv)["label"].value_counts().sort_index()
    n0, n1 = int(counts[0]), int(counts[1])
    print("n0:", n0, "  n1:", n1)
    n = n0 + n1
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    print("w0: ", w0, "w1: ", w1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    class_weights = class_weights / class_weights.max()
    print("w: ", class_weights.tolist())
    class_weights=class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("weights:", class_weights.tolist())
    return class_weights, criterion

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights,criterion=compute_class_weights_and_loss(jf["root"]/jf["train.csv"],device)'''



'''ckpt=torch.load("best.pt",map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss,test_acc=evaluate_one_epoch(model,test_loader,criterion, device)

    print(f"TEST:loss={test_loss:.4f}, acc={test_acc:.4f}")'''


'''   if __name__=="__main__":
       print("Start************")
       main()'''

'''
cap=cv2.VideoCapture(0)
detector=LeftEyeLandmarkDetector()
saved=0
max_saved=5
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    eye_img, box=detector.detect_left_eye(frame)

    if eye_img is not None:
        cv2.imshow("left eye",eye_img)
        x,y,w,h=box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

        if saved < max_saved:
            filename = f"left_eye_{saved + 1:02d}.png"
            cv2.imwrite(filename, eye_img)
            print(f" saved  done : {filename}")
            saved += 1
        if saved >= max_saved:
            print(" 10 frames are saved.")
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()'''