import torch
from PIL import Image

import config
from model import eval_transform, load_trained


class Predictor:
    def __init__(self, spec=None, threshold=config.CONF_THRESHOLD, device="cpu"):
        spec = spec or config.ACTIVE_MODEL
        self.device = device
        self.threshold = threshold
        self.model, self.classes = load_trained(spec["arch"], spec["weights"], device)
        self.transform = eval_transform(spec["img_size"])

    @torch.no_grad()
    def predict(self, image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        top_p, top_i = probs.topk(min(3, len(self.classes)))
        return {
            "label": self.classes[idx.item()],
            "confidence": conf.item(),
            "accepted": conf.item() >= self.threshold,
            "top3": [(self.classes[i], p) for i, p in zip(top_i.tolist(), top_p.tolist())],
        }

    def predict_path(self, path):
        with Image.open(path) as img:
            return self.predict(img)