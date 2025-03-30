import os
import cv2
import xml.etree.ElementTree as ET
from datetime import timedelta
from torch.utils.data import Dataset

def time_to_seconds(t: str) -> float:
    h, m, s = t.split(':')
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s)).total_seconds()

class FightEventDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_interval=30):
        self.samples = []
        self.transform = transform
        self.frame_interval = frame_interval  # 추가됨

        for file in os.listdir(root_dir):
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename = root.find("filename").text
                video_path = os.path.join(root_dir, filename)
                fps = int(root.find("header/fps").text)
                xml_total_frames = int(root.find("header/frames").text)

                cap = cv2.VideoCapture(video_path)
                actual_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                total_frames = min(xml_total_frames, actual_total_frames)

                event = root.find("event")
                eventname = event.find("eventname").text
                start = time_to_seconds(event.find("starttime").text)
                duration = time_to_seconds(event.find("duration").text)

                start_frame = int(start * fps)
                end_frame = int((start + duration) * fps)

                # 🎯 이벤트 프레임 (frame_interval 단위로)
                for f in range(start_frame, min(end_frame + 1, total_frames), self.frame_interval):
                    self.samples.append((video_path, f, eventname))

                # 🎯 일반 프레임 (frame_interval 단위로)
                for f in range(0, min(start_frame, total_frames), self.frame_interval):
                    self.samples.append((video_path, f, "general"))
                for f in range(end_frame + 1, total_frames, self.frame_interval):
                    self.samples.append((video_path, f, "general"))

        self.samples.sort(key=lambda x: (x[0], x[1]))  # 파일명 + 프레임 기준 정렬

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_id, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"Failed to read frame {frame_id} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)

        return frame, label
