import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
import time
# from tqdm.auto import tqdm

from util.ImageUtil import ImageReader, ImageWriter
from models.matting.MattingNetwork import MattingNetwork


"""调用方法：
先实例化Matting，模型文件已经配置好，运行convert进行转换"""
class Matting:
    def __init__(self, variant: str, model_path: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

    def convert(self, *args, **kwargs):
        human_matting(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

def human_matting(model,
                  input_source:str = "./for_test/images",
                  input_resize:Optional[Tuple[int,int]] = None,
                  out_path = "./for_test/out", 
                  device:Optional[str] = "cuda", 
                  dtype: Optional[torch.dtype] = None

                  ):
    # initial series: inference dataloader,models, writers
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    source = ImageReader(input_source, transform)
    
    reader = DataLoader(source, 
                        batch_size=1, 
                        pin_memory=True, 
                        num_workers=1)

    writer = ImageWriter(out_path, "png")

    model = model.eval()
    if dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
    
    # inference
    try:
        with torch.no_grad():
            # bar = tqdm
            rec = [None] * 4
            for src in reader:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) #[BTCHW]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                #fgr for foreground, pha for alpha
                fgr = fgr * pha.gt(0)
                com = torch.cat([fgr, pha], dim=-3)
                writer.write(com[0])

    finally:
        writer.close()

if __name__ == "__main__":

    converter = Matting("mobilenetv3", r"./enjoyPainter/tools/models/matting/rvm_mobilenetv3.pth", "cuda")
    start_time = time.time()
    converter.convert(
        input_source="./for_test/images",
        out_path="./temp"
    )
    end_time = time.time()
    # all_time = 1
    # count = 50
    # for _ in range(count): # 测个时间
    #     st_time = time.time()
    #     converter.convert(
    #         input_source="../for_test/images",
    #         out_path="./temp"
    #     )
    #     end_time = time.time()
    #     all_time += end_time-st_time
    print(f"图片转换完成，共用时{end_time - start_time}秒")