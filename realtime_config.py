import json
from pathlib import Path
from xml.etree.ElementTree import indent


class RealtimeConfig:
    def __init__(self, rt_json_file):
        with open(rt_json_file, "r") as f:
            self.re_jf = json.load(f)
    def __getitem__(self, key):
        return self.re_jf[key]

    def show(self):
        print("----RealtimeConfig-----")
        for m, n in self.re_jf.items():
            print(f"{m}: {n}")
        print("-------------------")


