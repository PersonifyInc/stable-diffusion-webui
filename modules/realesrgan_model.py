import os

import numpy as np
from PIL import Image
from realesrgan import RealESRGANer

from modules.upscaler import Upscaler, UpscalerData
from modules.shared import cmd_opts, opts
from modules import modelloader, errors


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN"
        self.user_path = path
        super().__init__()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
            from realesrgan import RealESRGANer  # noqa: F401
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # noqa: F401
            self.enable = True
            self.scalers = []
            self.upsampler = None
            scalers = self.load_models(path)

            local_model_paths = self.find_models(ext_filter=[".pth"])
            for scaler in scalers:
                if scaler.local_data_path.startswith("http"):
                    filename = modelloader.friendly_name(scaler.local_data_path)
                    local_model_candidates = [local_model for local_model in local_model_paths if local_model.endswith(f"{filename}.pth")]
                    if local_model_candidates:
                        scaler.local_data_path = local_model_candidates[0]

                if scaler.name in opts.realesrgan_enabled_models:
                    self.scalers.append(scaler)

        except Exception:
            errors.report("Error importing Real-ESRGAN", exc_info=True)
            self.enable = False
            self.scalers = []

    def do_upscale(self, img, path):
        if not self.enable:
            return img

        try:
            info = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load RealESRGAN model {path}", exc_info=True)
            return img

        self.upsampler = RealESRGANer(
            scale=info.scale,
            model_path=info.local_data_path,
            model=info.model(),
            half=not cmd_opts.no_half and not cmd_opts.upcast_sampling,
            tile=opts.ESRGAN_tile,
            tile_pad=opts.ESRGAN_tile_overlap,
            device=self.device,
        )

        upsampled = self.upsampler.enhance(np.array(img), outscale=info.scale)[0]

        image = Image.fromarray(upsampled)
        return image

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"RealESRGAN data missing: {scaler.local_data_path}")
                return scaler
        raise ValueError(f"Unable to find model info: {path}")

    def load_models(self, _):
        return get_realesrgan_models(self)

    def release_model(self):
        from modules import devices
        del self.upsampler
        self.upsampler = None
        devices.torch_gc()

    def get_total_steps(self, width, height, name, scale):
        import math
        info = next(iter([scaler for scaler in self.scalers if scaler.name == name]), None)
        if (info is None):
            return 0

        total_steps = 0

        dest_w = int((width * scale) // 8 * 8)
        dest_h = int((height * scale) // 8 * 8)

        width = width + opts.ESRGAN_tile_overlap
        height = height + opts.ESRGAN_tile_overlap
        for _ in range(3):
            tiles_x = math.ceil(width / opts.ESRGAN_tile)
            tiles_y = math.ceil(height / opts.ESRGAN_tile)
            total_steps += tiles_x * tiles_y

            shape = (width, height)

            new_shape = (width * info.scale, height * info.scale)

            if shape == new_shape:
                break

            if new_shape[0] >= dest_w and new_shape[1] >= dest_h:
                break

            (width, height) = new_shape

        return total_steps
    
    def update_step_num(self, data):
        import re
        splits = re.split('\s+|Tile|/', data)
        splits = list(filter(bool, splits))
        splits = [int(e) for e in splits if e.isdigit()]
        if (len(splits) >= 2):
            return 1
        return 0

def get_realesrgan_models(scaler):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        models = [
            UpscalerData(
                name="R-ESRGAN General 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN General WDN 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN AnimeVideo",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN 4x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 4x+ Anime6B",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 2x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                scale=2,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        return models
    except Exception:
        errors.report("Error making Real-ESRGAN models list", exc_info=True)
