from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
from torch.utils.data import Dataset


class FileTypeError(Exception):
    def __init__(self, error_message = "file type error!"):
        self.message = error_message
        super().__init__(self.message)

class ImageReader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.file_type = 0 # 0 for file path, 1 for file
        try:
            if os.path.isfile(self.path):
                if not (self.path.endswith(".png") or self.path.endswith(".jpg")):
                    raise FileTypeError("input file type should be jpg or png!")
                else:
                    self.file_type = 1
        except:
            pass  
        self.files = sorted(os.listdir(path)) if self.file_type == 0 else 1

        self.transform = transform
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.path
        if self.file_type == 0:
            img_path = os.path.join(self.path, self.files[idx])
        with Image.open(img_path) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img
        

class ImageWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1
            
    def close(self):
        pass