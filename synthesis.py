import os
import torch
from PIL import Image
from torchvision import transforms
from models.cyclegan_modules import Generator

# Configuration
MRI_DIR = "./data/mri"
CT_DIR = "./data/ct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_ct():
    netG = Generator().to(DEVICE)
    # Load your trained weights (or initialize for testing)
    # netG.load_state_dict(torch.load("weights/gen_mri2ct.pth"))
    netG.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for split in ['train', 'test']:
        for cat in ['glioma', 'meningioma', 'pituitary', 'notumor']:
            path = os.path.join(MRI_DIR, split, cat)
            out_path = os.path.join(CT_DIR, split, cat)
            os.makedirs(out_path, exist_ok=True)
            
            for img_name in os.listdir(path):
                img = Image.open(os.path.join(path, img_name)).convert("L")
                input_t = transform(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    fake_ct = netG(input_t)
                    # Convert back to image
                    fake_ct = (fake_ct.squeeze().cpu().numpy() + 1) / 2.0 * 255
                    Image.fromarray(fake_ct.astype('uint8')).save(os.path.join(out_path, img_name))

if __name__ == "__main__":
    generate_ct()
    print("Synthetic CT scans generated successfully.")