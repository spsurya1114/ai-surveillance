
import torch
import cv2
import torchvision.models as models

try:
    import torchreid
    HAS_TORCHREID = True
except ImportError:
    HAS_TORCHREID = False


class ReID:
    def __init__(self):
        # Set device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if HAS_TORCHREID:
            try:
                # Load model
                self.model = torchreid.models.build_model(
                    name='osnet_x1_0',
                    num_classes=1000,
                    pretrained=True
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_fallback = False
                return
            except Exception as e:
                print(f"Failed to load torchreid model, falling back to torchvision: {e}")

        # Fallback to pretrained MobileNetV3 Small for feature extraction
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove classifier head by setting to identity, yielding the pooled feature representation
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        self.is_fallback = True

    def extract_features(self, image):
        # Resize image
        img = cv2.resize(image, (128, 256))

        # Convert to tensor
        img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Move to device
        img = img.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img)

        return features.squeeze().cpu().numpy()


