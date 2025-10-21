import cv2, json, os

class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.idx=0

    def __iter__(self): return self
    def __next__(self):
        ok, frame = self.cap.read()
        if not ok: 
            raise StopIteration
        self.idx+=1
        return self.idx - 1,frame
    def close(self): self.cap.release()

class VideoWriter:
    def __init__(self, out_dir, reader, enabled=True):
        self.enabled = enabled
        if enabled:
            path = os.path.join(out_dir, "annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.vw = cv2.VideoWriter(path, fourcc, reader.FPS, (reader.W, reader.H))
    def write(self, frame): 
        if self.enabled: self.vw.write(frame)
    def close(self):
        if self.enabled: self.vw.release()

class JsonlWriter:
    def __init__(self, path): self.f = open(path,"w",encoding="utf-8")
    def write(self, obj): self.f.write(json.dumps(obj)+"\n")
    def close(self): self.f.close()
