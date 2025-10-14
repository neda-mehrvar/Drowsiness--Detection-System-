import json
from pathlib import Path
class Config:
    def __init__(self,json_file):
        with open (json_file,"r") as f:
            self.jf=json.load(f)
        self.jf["root"]=Path(self.jf["root"])
    def __getitem__(self,key):
        return self.jf[key]
    def show(self):
        print("----Config-----")
        for m,n in self.jf.items():
            print(f"{m}: {n}")
        print("-------------------")




