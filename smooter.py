import time
from collections import deque

class ProbSmoother:
    def __init__(self,window_sec: float, thresh: float, main_alert_sec: float):
        self.window_sec=float(window_sec)
        self.thresh=float(thresh)
        self.main_alert_sec=float(main_alert_sec)
        self.buf:deque[tuple[float,float]]=deque()
        self._over_thresh_since=None
    def update(self,prob: float)-> tuple[float,bool]:
        now=time.time()
        self.buf.append((now,float(prob)))
        cutoff=now - self.window_sec
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()
        if not self.buf:
            return 0.0 , False
        avg=sum(p for _, p in self.buf)/ len(self.buf)
        if avg > self.thresh:
            if self._over_thresh_since is None :
                self._over_thresh_since=now
            over_dur=now- self._over_thresh_since
            alarm=over_dur >= self.main_alert_sec
        else:
            self._over_thresh_since=None
            alarm=False
        return avg, alarm

