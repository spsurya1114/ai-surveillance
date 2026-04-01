
import torchreid
import torch
import cv2


class ReID:
    def __init__(self):
        # Set device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

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

