import base64
import io
import os
import time
import datetime
import uvicorn
import ipaddress
import requests
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart, shared_items
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
from modules.sd_models import unload_model_weights, reload_model_weights, checkpoint_aliases, checkpoints_path_list, list_models
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import Dict, List, Any
import piexif
import piexif.helper
from contextlib import closing
import os
import numpy as np
import whisper
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoProcessor,
)

import threading
import ctypes
import time
import torch
import psutil
from modules import devices
import subprocess as sp
import traceback
import sys
import errno
from sentence_transformers import SentenceTransformer, util
import json
import modules.watermark as waterm
import os
from multiprocessing import Pool
import hmac, hashlib, json
import boto3
import shutil
from modules_aura.Scripts.filterWord import filterWord
from enum import Enum
import math
from dotenv import load_dotenv

load_dotenv()

def estimateTime(w, h, s, up, upS, hires):
    def genF( w, h, s):
        return ((s/20) * 1.6 * (w*h) / (512*512))

    def upScale( w, h):
        return (1.7 * (w*h) / (512*512))

    def hiresF( w, h, s, up):
        s = max(s, 20)
        up = max(up, 2)
        if w != h:
            # return upScale(w, h)*2 + (s/20) * (genF(w*up, h*up, s)) * (up/2) * math.sqrt((w*h) / (512*512))
            return upScale(w, h)*2 + (genF(w*up, h*up, s)) * (up/2) * math.sqrt((w*h) / (512*512))
        else:
            # return (s/20) * genF(w*up, h*up, s) * (up/2) * math.sqrt((w*h) / (512*512))
            return genF(w*up, h*up, s) * (up/2) * math.sqrt((w*h) / (512*512))

    if hires:
        return hiresF( w, h, s, up)
    elif upS and up > 1:
        return upScale(w, h)
    else:
        return genF(w, h, s)

class ErrorCode(Enum):
    OK = 0
    ErrOOM = 1
    ErrSpace = 2
    ErrData = 3
    ErrPrompt = 4
    Undefined = 99

class thread_with_exception(threading.Thread):
    def __init__(self, name, func, processor):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.processor = processor
        self.finished = False
        self.result = None
        self.error_code = ErrorCode.OK.value
        self.error = None
            
    def run(self):
        # target function of the thread class
        try:
            try:
                self.result = self.func(self.processor)
            except RuntimeError as rEx:
                print("Runtime Error occured: " + str(rEx))
                traceback.print_exc()
                self.error = rEx
                if (isinstance(rEx, torch.cuda.OutOfMemoryError)):
                    self.error_code = ErrorCode.ErrOOM.value
            except Exception as ex:
                print("Error occured: " + str(ex))
                traceback.print_exc()
                self.error = ex
                self.error_code = ErrorCode.OK.value
        finally:
            print("Finish processing thread for " + self.name)
            self.processor = None
            devices.torch_gc()
            self.finished = True
        
    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
            ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

from contextlib import redirect_stdout
class WriteProcessor:
    def __init__(self, update_func = None):
        self.buf = ""
        self.real_stdout = sys.stdout
        self.update_func = update_func

    def write(self, buf):
        # emit on each newline
        while buf:
            try:
                newline_index = buf.index("\n")
            except ValueError:
                # no newline, buffer for next call
                self.buf += buf
                break
            # get data to next newline and combine with any buffered data
            data = self.buf + buf[:newline_index + 1]
            self.buf = ""
            buf = buf[newline_index + 1:]

            self.real_stdout.write(data)
            self.real_stdout.flush()
            if (self.update_func is not None):
                self.update_func(data)

def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in shared.sd_upscalers])}") from e

def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def verify_url(url):
    """Returns True if the url refers to a global resource."""

    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True


def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        response = requests.get(encoding, timeout=30, headers=headers)
        try:
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e

def compute_signature(uri, date='', body='', serectkey=None):
    if isinstance(body, dict):
        body = json.dumps(body)

    if serectkey is None:
        serectkey = os.getenv("CLOUD_KEYBASE_SECRET")
    to_sign = uri + body + date
    sig = hmac.new(bytes(serectkey, 'utf-8'), bytes(to_sign, 'utf-8'), hashlib.sha1)

    return base64.b64encode(sig.digest())

def upFileToS3(file_name, user_id):

    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param s3Path: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # Upload the file
    s3_client = boto3.client('s3',
                            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("AWS_ACCESS_SECRET"))
    bucket = os.getenv("S3_BUCKET")
    s3Path = os.getenv("S3_STORAGE_PATH") + "/" + user_id + "/" + os.path.basename(file_name)
    try:
        response = s3_client.upload_file(file_name, bucket, s3Path)
    except Exception as e:
        print("Exception", str(e))
        raise e
    return True

def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console
            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.initDone = False
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.write_lock = threading.Lock()
        api_middleware(self.app)
        # self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=models.TextToImageResponse)
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapiCloud, methods=["POST"]) # text2imgapiCloud text2imgapi
        self.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"], response_model=models.ImageToImageResponse)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ExtrasSingleImageResponse)
        # self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ExtrasBatchImagesResponse)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_apiCloud, methods=["POST"]) # extras_batch_images_apiCloud
        self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=models.PNGInfoResponse)
        self.add_api_route("/sdapi/v1/progress", self.getprogressapiCloud, methods=["GET"], response_model=models.ProgressResponse) # getprogressapiCloud getprogressapi
        self.add_api_route("/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapiCloud, methods=["POST"]) # interruptapiCloud interruptapi
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=List[models.SamplerItem])
        self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=List[models.UpscalerItem])
        self.add_api_route("/sdapi/v1/latent-upscale-modes", self.get_latent_upscale_modes, methods=["GET"], response_model=List[models.LatentUpscalerModeItem])
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[models.SDModelItem])
        self.add_api_route("/sdapi/v1/sd-vae", self.get_sd_vaes, methods=["GET"], response_model=List[models.SDVaeItem])
        self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[models.HypernetworkItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[models.FaceRestorerItem])
        self.add_api_route("/sdapi/v1/realesrgan-models", self.get_realesrgan_models, methods=["GET"], response_model=List[models.RealesrganItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=List[models.PromptStyleItem])
        self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-vae", self.refresh_vae, methods=["POST"])
        self.add_api_route("/sdapi/v1/create/embedding", self.create_embedding, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/create/hypernetwork", self.create_hypernetwork, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/preprocess", self.preprocess, methods=["POST"], response_model=models.PreprocessResponse)
        self.add_api_route("/sdapi/v1/train/embedding", self.train_embedding, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/train/hypernetwork", self.train_hypernetwork, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=List[models.ScriptInfo])
        self.add_api_route("/sdapi/v1/getqueue", self.getQueue, methods=["GET"])

        if shared.cmd_opts.api_server_stop:
            self.add_api_route("/sdapi/v1/server-kill", self.kill_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-restart", self.restart_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []
        self.uuid_lst = []
        self.uuid = None
        self.user_id = None
        self.imgQueue = 0
        self.estimetime = 0
        self.uuid_download_lst = []

        self.add_api_route("/sdapi/v1/audio2img", self.audio2imgapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/audio2txt", self.audio2txtapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/upload", self.upload, methods=["POST"])

        self.speech_mode_paths = {
            "en": "models/models--openai--whisper-tiny.en/"
        }

        import torch
        self.device = "cuda"
        self.speechmodel = None
        self.Audioprocessor = None
        self.sentence2embed = None
        # if (torch.cuda.is_available()):
        #     self.speechmodel = WhisperForConditionalGeneration.from_pretrained("models/models--openai--whisper-tiny.en/").to(self.device)
        # else:
        #     self.speechmodel = WhisperForConditionalGeneration.from_pretrained("models/models--openai--whisper-tiny.en/")
        # self.Audioprocessor = AutoProcessor.from_pretrained("models/models--openai--whisper-tiny.en/")

        model_path = 'models/sentence-transformers_all-MiniLM-L6-v2'
        if (torch.cuda.is_available()):
            self.sentence2embed = SentenceTransformer(model_path, device=self.device)
        else:
            self.sentence2embed = SentenceTransformer(model_path)

        if (torch.cuda.is_available()):
            torch.cuda.set_per_process_memory_fraction(shared.cmd_opts.gpu_memory_fraction)

        self.watermark = waterm.EmbedDwtDctSvd()
        self.filterBadWord = filterWord(r"modules_aura/Models/TextToVector/clip-vit-large-patch14")
        self.filterBadWord.filterWordPath(r"modules_aura/Data/ForbiddenWords")
        self.initDone = True

    def release_speech_models(self):
        if (self.speechmodel is not None):
            del self.speechmodel
            self.speechmodel = None
        if (self.Audioprocessor is not None):
            del self.Audioprocessor
            self.Audioprocessor = None
        devices.torch_gc()

    def init_speech_models(self, language="en"):
        model_path = "models/models--openai--whisper-tiny.en/"
        if (language in self.speech_mode_paths):
            model_path = self.speech_mode_paths[language]
        if (self.speechmodel is None):
            if (torch.cuda.is_available()):
                self.speechmodel = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            else:
                self.speechmodel = WhisperForConditionalGeneration.from_pretrained(model_path)
        if (self.Audioprocessor is None):
            self.Audioprocessor = AutoProcessor.from_pretrained("models/models--openai--whisper-tiny.en/")


    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script_info(self):
        res = []

        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts]:
            res += [script.api_info for script in script_list if script.api_info is not None]

        return res

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        #find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # get default values
        with gr.Blocks(): # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts:
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        return script_args

    def upload(self, inputs: dict):
        print(inputs.keys())
        base64_message = inputs['filedata']
        base64_bytes = base64_message.encode('utf-8')
        message_bytes = base64.b64decode(base64_bytes)
        with open(inputs['filename'], 'wb') as f:
            f.write(message_bytes)

        return {"result": "OKIE"}

    def Speech2txt(self, input_speech):
        # print(input_speech)
        input_speech = np.array(input_speech, dtype=float)
        print("Speech2txt", input_speech.shape)

        with self.queue_lock:
            self.init_speech_models()
            inputs = self.Audioprocessor(input_speech, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            generated_ids = self.speechmodel.generate(inputs=input_features)

            transcription = self.Audioprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.release_speech_models()

        return transcription

    def getQueue(self):
        return {"imgqueue": self.imgQueue, "estimetime": self.estimetime, "uuid_lst":self.uuid_lst}

    def audio2txtapi(self, inputs: dict):
        # print("audio2imgapi", inputs)
        
        import os
        fileName = os.path.basename(inputs['filename'])
        with open(fileName, 'wb') as f:
            base64_message = inputs['filedata']
            base64_bytes = base64_message.encode('utf-8')
            message_bytes = base64.b64decode(base64_bytes)
            print(message_bytes[:10])
            f.write(message_bytes)

        waveData = whisper.load_audio(fileName)
        txt = self.Speech2txt(waveData)
        print("txt output: ", txt)

        return {"speech_text": txt}

    def audio2imgapi(self, inputs: dict):
        # print("audio2imgapi", inputs)
        
        import os
        fileName = os.path.basename(inputs['filename'])
        with open(fileName, 'wb') as f:
            base64_message = inputs['filedata']
            base64_bytes = base64_message.encode('utf-8')
            message_bytes = base64.b64decode(base64_bytes)
            print(message_bytes[:10])
            f.write(message_bytes)

        inputs.pop("filename", None)
        inputs.pop("filedata", None)

        waveData = whisper.load_audio(fileName)
        txt = self.Speech2txt(waveData)
        print("txt output: ", txt)
        inputs["prompt"] = txt
        inputs["text2embeds"] = txt

        print(inputs)
        args = models.StableDiffusionTxt2ImgProcessingAPI(**inputs)
        result = self.text2imgapi(args)
        result.speech_text = txt

        return result

    def generation_oom_process_handle(self, process_name, p, process_func, free_ratio_allow, oom_time_out_seconds, 
        em_time_out_seconds, check_oom_interval_seconds):

        def get_gpu_memory():
            mem_free = 0
            memory_free_values = {}
            try:
                command = "nvidia-smi --query-gpu=index,memory.free --format=csv"
                memory_free_info = sp.check_output(command.split()).decode('ascii')
                memory_free_info = memory_free_info.replace('\r\n', '\n').replace('\r', '\n').replace(' ', ',')
                memory_free_info = memory_free_info.split('\n')[:-1][1:]
                for info in memory_free_info:
                    items = info.split(",")
                    items = [x for x in items if x]
                    memory_free_values[float(items[0])] = float(items[1])
                cuda_device_id = torch.cuda.current_device()
                if (cuda_device_id in memory_free_values):
                    mem_free = memory_free_values[cuda_device_id]
            except:
                pass
            return mem_free 
        
        result = None
        error_code = ErrorCode.Undefined.value
        try:      
            genThread = thread_with_exception(process_name, process_func, p)
            genThread.start()
                
            oomCount = 0
            emTimeOutCount = 0
            emTimeOut = em_time_out_seconds
            exhaustLimitMbs = 200
            oomTimeOutCount = oom_time_out_seconds
            oomTimeOutBeginning = oom_time_out_seconds
            oomTimeOutProcessing = oom_time_out_seconds * 1.5
            oomTimeOutEnding = oom_time_out_seconds * 2
            prevStep = 0
            isResetCountAtEnd = False
            while True:
                try:
                    if shared.state.interrupted:
                        print("generation_oom_process_handle interrupting")
                        genThread.error_code = ErrorCode.OK.value
                        genThread.raise_exception()
                        genThread.join()
                        break

                    isOOM = False
                    if (torch.cuda.is_available()):
                        free = get_gpu_memory()
                        free, total = torch.cuda.mem_get_info()
                        if (free == 0):
                            free = get_gpu_memory()
                    else:
                        free = psutil.virtual_memory().free
                        total = psutil.virtual_memory().total

                    freeMbs = free / 1024 / 1024
                    totalMbs = total / 1024 / 1024
                    avaiable_ratio = free / total
                    total_steps = shared.total_tqdm.get_total_steps() 
                    current_step = shared.total_tqdm.step
                    if (avaiable_ratio < free_ratio_allow):
                        isOOM = True

                    if (current_step <= prevStep and isOOM):
                        oomCount += 1
                    else:
                        emTimeOutCount = 0
                        oomCount = 0
                    
                    if (total_steps > 0 and current_step == total_steps):
                        if (isResetCountAtEnd == False):
                            emTimeOutCount = 0
                            oomCount = 0
                            isResetCountAtEnd = True

                    if (current_step == 0): 
                        oomTimeOutCount = oomTimeOutBeginning
                    elif (current_step > 0 and current_step < total_steps):
                        oomTimeOutCount = oomTimeOutProcessing
                    elif (current_step >= total_steps):
                        oomTimeOutCount = oomTimeOutEnding

                    if (current_step == 0 and freeMbs < exhaustLimitMbs):
                        emTimeOutCount += 1

                    if isOOM:
                        print(f"generation_oom_process_handle Out of memory warning ==> Memory: {free}, {freeMbs:.8f}/{totalMbs:.2f}->{(avaiable_ratio*100):.2f}%" +                      
                            f", Out of memory count: {oomCount}/{oomTimeOutCount}"+
                            f", Exhaust memory count: {emTimeOutCount}/{emTimeOut}")
                        
                    if (emTimeOutCount > emTimeOut or oomCount > oomTimeOutCount):
                        print("generation_oom_process_handle Out of memory occured, terminating generation process")
                        genThread.error_code = ErrorCode.ErrOOM.value
                        genThread.raise_exception()
                        genThread.join()
                        break

                    if (genThread.finished):
                        break

                    prevStep = current_step
                    time.sleep(check_oom_interval_seconds)
                except Exception as ex:
                    print("generation_oom_process_handle check memory loop occured error: " + str(ex))
                    traceback.print_exc()
                    genThread.error_code = ErrorCode.ErrSpace.value
                    genThread.raise_exception()
                    genThread.join()
                    break
            
            print("generation_oom_process_handle get out of check memory loop")
            print("generation_oom_process_handle setting result")
            if (genThread.error_code == ErrorCode.OK.value):
                result = genThread.result

            print("generation_oom_process_handle setting error code")
            error_code = ErrorCode.OK.value
            if (result is None or genThread.error_code > ErrorCode.OK.value):
                error_code = ErrorCode.Undefined.value if genThread.error_code == ErrorCode.OK.value else genThread.error_code
                if (shared.state.interrupted):
                    error_code = ErrorCode.OK.value
            print("generation_oom_process_handle setting result finish")
            shared.state.interrupted = False
        except Exception as ex:
            print("generation_oom_process_handle occured error: " + str(ex))
            traceback.print_exc()
        
        del genThread
        p = None
        devices.torch_gc()
        print("generation_oom_process_handle get out of generation_oom_process_handle")     
        return result, error_code

    def text2imgapiCloud(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        print("Processing text2imgapiCloud")
        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        uuid = args['uuid']

        if args['actual_prompt']:
            badWord = self.filterBadWord.detectBadWord(args['actual_prompt'])
            print("badWord", args['actual_prompt'], badWord)
            if badWord:
                return {
                    "status": "failed",
                    "uuid": uuid,
                    "error_message": badWord,
                    "error": ErrorCode.ErrPrompt.value,
                    }
        total_waiting = self.estimetime
        thread = threading.Thread(target=self.runProcess_txt2img, args=[txt2imgreq, args])
        thread.start()

        return {
            "status": "running",
            "uuid": uuid,
            "total_waiting": total_waiting,
        }

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        print("Processing text2imgapi", txt2imgreq)

        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        print(txt2imgreq.script_name, scripts.scripts_txt2img, txt2imgreq.sampler_name, txt2imgreq.sampler_index)
        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)
        devices.torch_gc()

        args["do_not_save_samples"] = True
        args["do_not_save_grid"] = True
        args["seed_resize_from_h"] = 0
        args["seed_resize_from_w"] = 0
        args["hr_upscaler"] = "Latent"
        args["denoising_strength"] = 0.7
        print("Args: " +  str(args))
        ar_width = args.pop('ar_width', 1)
        ar_height = args.pop('ar_height', 1)
        image_count = args.pop('image_count', None)
        args.pop('dimension_id', None)
        save_dir_path = args.pop('save_dir', None)
        send_paths = args.pop('send_paths', True)
        args.pop('alwayson_scripts', None)

        output_width = args["width"]
        output_height = args["height"]
        if (ar_width != 1 and ar_height != 1):
            args["width"] = 512
            args["height"] = 512
            args["rs_enable"] = True
            args["rs_upscaler"] = "Latent"
            args["rs_resize_mode"] = 1
            args["rs_denoising_strength"] = 0.8
            args["hr_sampler_name"] = args["sampler_name"]
            args["rs_sampler_name"] = args["sampler_name"]
            args["sampler_name"] = "Euler a"
            
            # 912 512
            if f'{ar_width}:{ar_height}' == '16:9': # f'{ar_width}:{ar_height}'
                args["rs_resize_x"] = 640
                args["rs_resize_y"] = 360
            elif f'{ar_width}:{ar_height}' == '4:3':
                args["rs_resize_x"] = 576
                args["rs_resize_y"] = 432
            elif f'{ar_width}:{ar_height}' == '3:4':
                args["rs_resize_x"] = 432 # 576:432 640:480
                args["rs_resize_y"] = 576
            elif f'{ar_width}:{ar_height}' == '3:2':
                args["rs_resize_x"] = 624 # 576:384 624:416
                args["rs_resize_y"] = 416
            elif f'{ar_width}:{ar_height}' == '2:3':
                args["rs_resize_x"] = 416
                args["rs_resize_y"] = 624

            output_width = args["rs_resize_x"]
            output_height = args["rs_resize_y"]
            args["rs_steps"] = args["steps"]
            args["steps"] = 20
            if args["enable_hr"]:
                # args["hr_resize_y"] = int(((args["rs_resize_y"] // 8 + 4) * 8)* args["hr_scale"])
                # args["hr_resize_x"] = int(args["rs_resize_x"] * args["hr_scale"])
                multi = (int(args["rs_resize_x"] * args["hr_scale"]) // (8*ar_width))
                args["hr_resize_x"] = int(multi * 8 * ar_width)
                args["hr_resize_y"] = int(multi * 8 * ar_height)
                args["hr_upscaler"] = "R-ESRGAN 4x+" # R-ESRGAN 4x+ SwinIR 4x
                args["denoising_strength"] = 0.7
                output_width = args["hr_resize_x"]
                output_height = args["hr_resize_y"]

        # support for upcale extras
        reqDict = {}
        extras_upscaling_total_steps = 0
        if args["extras_upscaling_resize"] and args["extras_upscaling_resize"] >= 1:
            reqDict = models.ExtrasSingleImageRequest()
            reqDict = setUpscalers(reqDict)
            reqDict['extras_upscaler_1'] = "R-ESRGAN 4x+" if reqDict["extras_upscaler_1"] == 'None' else reqDict["extras_upscaler_1"]
            reqDict['upscaling_resize'] = 2 if args["extras_upscaling_resize"] == 'None' else args["extras_upscaling_resize"]

            single_image_steps = postprocessing.scripts.scripts_postproc.get_total_steps(output_width, output_height, reqDict['extras_upscaler_1'], reqDict['upscaling_resize'])
            extras_upscaling_total_steps = single_image_steps * image_count
            args["extras_additional_steps"] = extras_upscaling_total_steps
            print(f"ExtrasUpscaleImageRequest: {str(reqDict)}, Single image steps: {single_image_steps}, Total images steps: {extras_upscaling_total_steps}")          

        reqDict.pop("image", None)
        args.pop('extras_upscaler_1', None)
        args.pop('extras_upscaling_resize', None)

        text2embeds = None
        processed = None
        with self.queue_lock:
            if args["text2embeds"] and len(args["text2embeds"]) > 1:

                text2embeds = self.sentence2embed.encode(args["text2embeds"])
                text2embeds = json.dumps(text2embeds.tolist())

            args.pop("text2embeds", None)

            with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                try:
                    shared.state.begin(job="scripts_txt2img")
                    def process_generation(sp):
                        try:
                            sp.is_api = True
                            sp.scripts = script_runner
                            sp.outpath_grids = opts.outdir_txt2img_grids
                            sp.outpath_samples = opts.outdir_txt2img_samples

                            if selectable_scripts is not None:
                                sp.script_args = script_args
                                processed = scripts.scripts_txt2img.run(sp, *sp.script_args) # Need to pass args as list here
                            else:
                                sp.script_args = tuple(script_args) # Need to pass args as tuple here
                                processed = process_images(sp)
                        except Exception as ex:
                            raise ex
                        return processed

                    processed, error_code = self.generation_oom_process_handle("text2imgapi", p,
                        process_generation, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                        shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)

                except Exception as ex:
                    raise ex

            devices.torch_gc()

            if reqDict and error_code == ErrorCode.OK.value and processed is not None and shared.state.interrupted == False:
                def extras_upscale_process():
                    if (shared.state.interrupted):
                        return None, 0

                    def update_step_func(data):
                        update_steps = postprocessing.scripts.scripts_postproc.update_step_num(data)
                        if (update_steps > 0):
                            for _ in range(update_steps):
                                shared.total_tqdm.update()
                    print_capture = WriteProcessor(update_step_func)

                    p = postprocessing
                    def process_upscale(p):
                        try:
                            with redirect_stdout(print_capture):
                                result = p.run_extras(extras_mode=1, image_folder=processed.images, image="", input_dir="", output_dir="", save_output=False, **reqDict)
                        except Exception as ex:
                            raise ex
                        return result

                    if (shared.state.interrupted):
                        return None, 0

                    result, error_code = self.generation_oom_process_handle("extras_upscaling_resize", p,
                        process_upscale, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                        shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)
                    postprocessing.scripts.scripts_postproc.image_changed()
                    postprocessing.scripts.scripts_postproc.release_model()
                    devices.torch_gc()

                    if (shared.state.interrupted):
                        return None, 0

                    images = None
                    if (isinstance(result, tuple) and len(result) >= 1):
                        images = result[0]
                    return images, error_code

                result, error_code = extras_upscale_process()
                processed.images = result

            devices.torch_gc()
            shared.total_tqdm.clear()
            shared.state.end()

        info = ""
        b64images = []
        image_paths = []
        if (processed is not None and shared.state.interrupted == False):
            info = processed.js()
            if (send_images and isinstance(processed.images, list)):
                if (send_paths):
                    with Pool(8) as pool:
                        processed.images = pool.map(self.watermark.encode, [processed.images[i] for i in range(0,len(processed.images))])

                    for image in processed.images:
                        if (shared.state.interrupted):
                            break
                        try:
                            path, _ = images.save_image(image, save_dir_path, basename="")
                            image_paths.append(path)
                        except Exception as e:
                            if hasattr(e, 'errno'):
                                if e.errno == errno.ENOSPC:
                                    print("Not enough disk space for saving images")
                                    error_code = ErrorCode.ErrSpace.value
                                    break
                                else:
                                    print(str(e))
                                    raise e 
                            else:
                                print(str(e))
                                raise e
                else:
                    with Pool(8) as pool:
                        processed.images = pool.map(self.watermark.encode, [processed.images[i] for i in range(0,len(processed.images))])
                    # for i in range(len(processed.images)):
                    #     wm, sc = self.watermark.decode(processed.images[i])
                    #     print(sc)
                    b64images = list(map(encode_pil_to_base64, processed.images))
        if (shared.state.interrupted):
            for path in image_paths:
                try:
                    os.remove(path)
                except Exception as ex:
                    print(f"Remove file {path} for interrupt error: {str(ex)}")
            image_paths = []
            b64images = []

        if (shared.state.interrupted == False and len(image_paths) == 0 and len(b64images) == 0 and error_code == ErrorCode.OK.value):
            error_code = ErrorCode.ErrData.value

        return models.TextToImageResponse(images=b64images, image_paths=image_paths, sentence_vector=text2embeds, parameters=vars(txt2imgreq), info=info, error=error_code)

    def callbackCS(self, uri, uuid, user_id, image_names, error_code, number_images, run_times=0, run_jobs=0, text2embeds = None):
        import requests, json
        if(error_code == ErrorCode.OK.value):
            payload = {
                "uuid": uuid,
                "user_id": user_id,
                "status": "done",
                "filenames": image_names,
                "sentence_vector": text2embeds,
                "error_code": error_code,
                "run_times": run_times,
                "run_jobs": run_jobs,
                "number_images": number_images,
            }
        else:
            payload = {
                "uuid": uuid,
                "user_id": user_id,
                "status": "failed",
                "error_code": error_code,
                "run_times": 0,
                "run_jobs": 0,
                "error_message": "",
            }

        date = str(datetime.datetime.now())
        print("callbackCS", date, "error_code", error_code)

        try:
            sig = compute_signature(uri, date=date, body=payload, serectkey=None).decode("utf-8")
            cloud_keybase = os.getenv("CLOUD_KEYBASE")
            header = {
                'Date': date,
                "Content-Type": "application/json",
                'Authorization': f"PSY {cloud_keybase}:{sig}",
            }

            cloud_url = os.getenv("CLOUD_HOST") + uri
            response = requests.post(url=cloud_url, data=json.dumps(payload), headers=header)
            print(uuid, user_id, response)
        except Exception as ex:
            print(str(ex))
            HTTPException(status_code=404, detail="err " + str(ex))

        return

    def runProcess_txt2img(self, txt2imgreq, args):
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        args.pop('actual_prompt', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        args.pop('save_images', None)
        devices.torch_gc()

        args["do_not_save_samples"] = True
        args["do_not_save_grid"] = True
        args["seed_resize_from_h"] = 0
        args["seed_resize_from_w"] = 0
        args["hr_upscaler"] = "Latent"
        args["denoising_strength"] = 0.7

        uuid = args['uuid']
        ar_width = args.pop('ar_width', 1)
        ar_height = args.pop('ar_height', 1)
        image_count = args.pop('image_count', None)
        args.pop('dimension_id', None)
        run_times = args.pop('run_times', None)
        save_dir_path = args.pop('save_dir', None)
        send_paths = args.pop('send_paths', True)
        send_images = args.pop('send_images', True)

        output_width = args["rs_resize_x"]
        output_height = args["rs_resize_y"]
        if (ar_width != 1 and ar_height != 1):
            args["width"] = 512
            args["height"] = 512
            args["rs_enable"] = True
            args["rs_upscaler"] = "Latent"
            args["rs_resize_mode"] = 1
            args["rs_denoising_strength"] = 0.8
            args["hr_sampler_name"] = args["sampler_name"]
            args["rs_sampler_name"] = args["sampler_name"]
            args["sampler_name"] = "Euler a"

            # 912 512
            if f'{ar_width}:{ar_height}' == '16:9': # f'{ar_width}:{ar_height}'
                args["rs_resize_x"] = 640
                args["rs_resize_y"] = 360
            elif f'{ar_width}:{ar_height}' == '4:3':
                args["rs_resize_x"] = 576
                args["rs_resize_y"] = 432
            elif f'{ar_width}:{ar_height}' == '3:4':
                args["rs_resize_x"] = 432 # 576:432 640:480
                args["rs_resize_y"] = 576
            elif f'{ar_width}:{ar_height}' == '3:2':
                args["rs_resize_x"] = 624 # 576:384 624:416
                args["rs_resize_y"] = 416
            elif f'{ar_width}:{ar_height}' == '2:3':
                args["rs_resize_x"] = 416
                args["rs_resize_y"] = 624

            output_width = args["rs_resize_x"]
            output_height = args["rs_resize_y"]
            args["rs_steps"] = args["steps"]
            args["steps"] = 20
            if args["enable_hr"]:
                # args["hr_resize_y"] = int(((args["rs_resize_y"] // 8 + 4) * 8)* args["hr_scale"])
                # args["hr_resize_x"] = int(args["rs_resize_x"] * args["hr_scale"])
                multi = (int(args["rs_resize_x"] * args["hr_scale"]) // (8*ar_width))
                args["hr_resize_x"] = int(multi * 8 * ar_width)
                args["hr_resize_y"] = int(multi * 8 * ar_height)
                args["hr_upscaler"] = "R-ESRGAN 4x+" # R-ESRGAN 4x+ SwinIR_4x
                args["denoising_strength"] = 0.7
                output_width = args["hr_resize_x"]
                output_height = args["hr_resize_y"]

        reqDict = {}
        extras_upscaling_total_steps = 0
        if args["extras_upscaling_resize"] and args["extras_upscaling_resize"] >= 1:
            reqDict = models.ExtrasSingleImageRequest()
            reqDict = setUpscalers(reqDict)
            reqDict['extras_upscaler_1'] = "R-ESRGAN 4x+" if reqDict["extras_upscaler_1"] == 'None' else reqDict["extras_upscaler_1"]
            reqDict['upscaling_resize'] = 2 if args["extras_upscaling_resize"] == 'None' else args["extras_upscaling_resize"]

            single_image_steps = postprocessing.scripts.scripts_postproc.get_total_steps(output_width, output_height, reqDict['extras_upscaler_1'], reqDict['upscaling_resize'])
            extras_upscaling_total_steps = single_image_steps * image_count
            args["extras_additional_steps"] = extras_upscaling_total_steps
            print(f" ExtrasUpscaleImageRequest: {str(reqDict)}, Single image steps: {single_image_steps}, Total images steps: {extras_upscaling_total_steps} ")          
        reqDict.pop("image", None)

        imgqueue = int(args['steps'] / 20) * args['batch_size'] * args['n_iter'] * (4 if args['enable_hr'] else 1) * int((args['width'] / 512)**2)
        estimetime = estimateTime(args['height'], args['width'], args['steps'],
                args['hr_scale'] if args['enable_hr'] else args['extras_upscaling_resize'], 
                args["extras_upscaling_resize"] and args["extras_upscaling_resize"] >= 1, args['enable_hr']) * args["batch_size"] * args["n_iter"]

        args.pop('extras_upscaler_1', None)
        args.pop('extras_upscaling_resize', None)

        print("Args: " +  str(args), estimetime, "=>", self.estimetime, run_times)

        if (sys.platform[:len('linux')] == 'linux'):
            save_dir_path = '/tmp/'

        with self.write_lock:
            self.imgQueue +=  imgqueue
            self.estimetime += estimetime
            self.uuid_lst.append(args['uuid'])

        with self.queue_lock:
            timeStart = time.time()
            self.uuid = args.pop('uuid', None)
            self.user_id = args.pop('user_id', None)
            with self.write_lock:
                if self.uuid not in self.uuid_lst:
                    self.imgQueue -=  imgqueue
                    self.estimetime -= estimetime
                    return

            text2embeds = None
            if args["text2embeds"] and len(args["text2embeds"]) > 1:
                text2embeds = self.sentence2embed.encode(args["text2embeds"])
                text2embeds = json.dumps(text2embeds.tolist())
            args.pop("text2embeds", None)

            with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                try:
                    shared.state.begin(job="scripts_txt2img")
                    def process_generation(sp):
                        try:
                            sp.is_api = True
                            sp.scripts = script_runner
                            sp.outpath_grids = opts.outdir_txt2img_grids
                            sp.outpath_samples = opts.outdir_txt2img_samples

                            if selectable_scripts is not None:
                                sp.script_args = script_args
                                processed = scripts.scripts_txt2img.run(sp, *sp.script_args) # Need to pass args as list here
                            else:
                                sp.script_args = tuple(script_args) # Need to pass args as tuple here
                                processed = process_images(sp)
                        except Exception as ex:
                            raise ex
                        return processed

                    processed, error_code = self.generation_oom_process_handle("text2imgapi", p,
                        process_generation, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                        shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)
                except Exception as ex:
                    raise ex

            devices.torch_gc()
            if reqDict and error_code == ErrorCode.OK.value and processed is not None and len(processed.images) > 0:
                def extras_upscale_process():
                    def update_step_func(data):
                        update_steps = postprocessing.scripts.scripts_postproc.update_step_num(data)
                        if (update_steps > 0):
                            for _ in range(update_steps):
                                shared.total_tqdm.update()
                    print_capture = WriteProcessor(update_step_func)

                    p = postprocessing
                    def process_upscale(p):
                        try:
                            with redirect_stdout(print_capture):
                                result = p.run_extras(extras_mode=1, image_folder=processed.images, image="", input_dir="", output_dir="", save_output=False, **reqDict)
                        except Exception as ex:
                            raise ex
                        return result

                    result, error_code = self.generation_oom_process_handle("extras_upscaling_resize", p,
                        process_upscale, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                        shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)
                    postprocessing.scripts.scripts_postproc.image_changed()
                    postprocessing.scripts.scripts_postproc.release_model()
                    devices.torch_gc()

                    images = None
                    if (isinstance(result, tuple) and len(result) >= 1):
                        images = result[0]
                    return images, error_code

                result, error_code = extras_upscale_process()
                processed.images = result

            shared.total_tqdm.clear()
            shared.state.end()

            uuid = self.uuid
            user_id = self.user_id
            run_times = time.time() - timeStart
        with self.write_lock:
            if uuid in self.uuid_lst:
                del self.uuid_lst[self.uuid_lst.index(uuid)]
                if uuid not in self.uuid_download_lst:
                    self.uuid_download_lst.append(uuid)
            self.imgQueue -=  imgqueue
            self.estimetime -= estimetime


        info = ""
        image_names = []
        if (processed is not None):
            info = processed.js()
            if (send_images and isinstance(processed.images, list) and len(processed.images)):
                try:
                    with Pool(8) as pool:
                        processed.images = pool.map(self.watermark.encode, [processed.images[i] for i in range(0,len(processed.images))])

                    for image in processed.images:
                        path, _ = images.save_image(image, os.path.join(save_dir_path, uuid), save_to_dirs=False, basename="")
                        image_names.append(os.path.basename(path))
                    shutil.make_archive(os.path.join(save_dir_path, uuid), 'zip', os.path.join(save_dir_path, uuid))
                    upFileToS3(os.path.join(save_dir_path, uuid)+'.zip', user_id)
                    os.remove(os.path.join(save_dir_path, uuid)+'.zip')
                    shutil.rmtree(os.path.join(save_dir_path, uuid))
                except Exception as e:
                    if hasattr(e, 'errno'):
                        if e.errno == errno.ENOSPC:
                            print("Not enough disk space for saving images")
                            error_code = ErrorCode.ErrSpace.value
                        else:
                            print(str(e))
                            error_code = ErrorCode.Undefined.value
                    else:
                        print(str(e))
                        error_code = ErrorCode.Undefined.value

        if len(image_names) == 0:
            error_code = ErrorCode.ErrData.value
        run_jobs = len(image_names) * (1 + (1 if args['enable_hr'] else 0) + (1 if args['rs_enable'] else 0))
        print(uuid, ",".join(image_names), run_times, run_jobs, error_code)
        self.callbackCS("/api/sdCallBack", uuid, user_id, ",".join(image_names), error_code, len(image_names), run_times, run_jobs, text2embeds)

        with self.write_lock:
            if uuid in self.uuid_download_lst:
                del self.uuid_download_lst[self.uuid_download_lst.index(uuid)]
        return

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop('script_name', None)
        args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
                p.init_images = [decode_base64_to_image(x) for x in init_images]
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_img2img_grids
                p.outpath_samples = opts.outdir_img2img_samples

                try:
                    shared.state.begin(job="scripts_img2img")
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args) # Need to pass args as tuple here
                        processed = process_images(p)
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    def runProcess_upscale(self, reqDict):
        save_dir_path = reqDict.pop('save_dir', None)
        send_paths = reqDict.pop('send_paths', True)
        reqDict.pop('image_count', None)
        run_times = reqDict.pop('run_times', None)
        image_list = reqDict.pop('imageList', [])
        image_folder = []
        total_steps = 0
        for x in image_list:
            image_data = None
            if os.path.exists(x.name) and not x.data:
                image_data = Image.open(os.path.abspath(x.name))
            elif x.data is not None:
                image_data = decode_base64_to_image(x.data)

            if image_data is not None:
                image_folder.append(image_data)
                total_steps += postprocessing.scripts.scripts_postproc.get_total_steps(image_data.width, image_data.height, reqDict['extras_upscaler_1'], reqDict['upscaling_resize'])

        def update_step_func(data):
            update_steps = postprocessing.scripts.scripts_postproc.update_step_num(data)
            if (update_steps > 0):
                for _ in range(update_steps):
                    shared.total_tqdm.update()

        print_capture = WriteProcessor(update_step_func)

        imgqueue = int(len(image_folder))
        estimetime = estimateTime(image_data.height, image_data.width, 20, reqDict['upscaling_resize'], True, False)

        with self.write_lock:
            self.estimetime += estimetime
            self.imgQueue +=  imgqueue
            self.uuid_lst.append(reqDict['uuid'])

        error_code = ErrorCode.OK.value
        devices.torch_gc()
        with self.queue_lock:
            timeStart = time.time()
            self.uuid = reqDict.pop('uuid', None)
            self.user_id = reqDict.pop('user_id', None)
            with self.write_lock:
                if self.uuid not in self.uuid_lst:
                    self.estimetime -= estimetime
                    self.imgQueue -=  imgqueue
                    return

            shared.state.begin(job="upscale")
            shared.total_tqdm.updateTotal(total_steps)
            p = postprocessing
            def process_upscale(p):
                try:
                    with redirect_stdout(print_capture):
                        result = p.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)
                except Exception as ex:
                    raise ex
                return result

            result, error_code = self.generation_oom_process_handle("extras_batch_images_api", p,
                process_upscale, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)

            shared.state.end()
            postprocessing.scripts.scripts_postproc.image_changed()
            postprocessing.scripts.scripts_postproc.release_model()
            devices.torch_gc()
            shared.total_tqdm.clear()

            uuid = self.uuid
            user_id = self.user_id
            run_times = time.time() - timeStart

        with self.write_lock:
            if uuid in self.uuid_lst:
                del self.uuid_lst[self.uuid_lst.index(uuid)]
                if uuid not in self.uuid_download_lst:
                    self.uuid_download_lst.append(uuid)

            self.imgQueue -=  imgqueue
            self.estimetime -= estimetime

        image_names = []
        if ( isinstance(result, tuple) and len(result) >= 1 and result[0] is not None):
            for image in result[0]:
                try:
                    image = self.watermark.encode(image)
                    path, _ = images.save_image(image, os.path.join(save_dir_path, uuid), save_to_dirs=False, basename="")
                    image_names.append(os.path.basename(path))
                    shutil.make_archive(os.path.join(save_dir_path, uuid), 'zip', os.path.join(save_dir_path, uuid))
                    # for image in image_paths:
                    # print(os.path.join(save_dir_path, uuid)+'.zip')
                    upFileToS3(os.path.join(save_dir_path, uuid)+'.zip', user_id)
                    os.remove(os.path.join(save_dir_path, uuid)+'.zip')
                    shutil.rmtree(os.path.join(save_dir_path, uuid))
                except Exception as e:
                    if hasattr(e, 'errno'):
                        if e.errno == errno.ENOSPC:
                            print("Not enough disk space for saving images")
                            error_code = ErrorCode.ErrSpace.value
                        else:
                            print(str(e))
                            error_code = ErrorCode.Undefined.value
                    else:
                        print(str(e))
                        error_code = ErrorCode.Undefined.value
                    break

        print(image_names, run_times)
        if len(image_names) == 0:
            error_code = ErrorCode.ErrData.value

        respond = self.callbackCS("/api/sdCallBack", uuid, user_id, ",".join(image_names), error_code, len(image_names), run_times, run_jobs=len(image_list))

        with self.write_lock:
            if uuid in self.uuid_download_lst:
                del self.uuid_download_lst[self.uuid_download_lst.index(uuid)]

        return respond

    def extras_batch_images_apiCloud(self, req: models.ExtrasBatchImagesRequest):
        print("Processing extras_batch_images_apiCloud")

        reqDict = setUpscalers(req)
        reqDict['extras_upscaler_1'] = "R-ESRGAN 4x+" if reqDict["extras_upscaler_1"] == 'None' else reqDict["extras_upscaler_1"]
        reqDict['upscaling_resize'] = 2 if reqDict["upscaling_resize"] == 'None' else reqDict["upscaling_resize"]

        for k,v in reqDict.items():
            if k != "imageList":
                print(k, v, end=" - ")
        total_waiting = self.estimetime
        print(total_waiting)
        uuid = reqDict['uuid']
        # print("Args: " +  str(reqDict))
        thread = threading.Thread(target=self.runProcess_upscale, args=[reqDict])
        thread.start()

        return {
            "status": "running",
            "uuid": uuid,
            "total_waiting": total_waiting,
        }

    def getprogressapiCloud(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py
        
        if req.uuid in self.uuid_download_lst:
            print("getprogressapiCloud ", "downloading", req, "current", self.uuid)
            return models.ProgressResponse(progress=0, eta_relative=0, uuid=req.uuid, status="downloading", state=shared.state.dict(), textinfo=shared.state.textinfo)

        if req.uuid not in self.uuid_lst or (shared.state.job_count == 0 and len(self.uuid_lst) == 0):
            print("getprogressapiCloud ", "undefined", req, "current", self.uuid)
            return models.ProgressResponse(progress=0, eta_relative=0, uuid=req.uuid, status="undefined", state=shared.state.dict(), textinfo=shared.state.textinfo)


        if req.uuid != self.uuid and req.uuid in self.uuid_lst:
            print("getprogressapiCloud ", "waiting", req, "current", self.uuid)
            return models.ProgressResponse(progress=0, eta_relative=0, uuid=req.uuid, status="waiting", state=shared.state.dict(), textinfo=shared.state.textinfo)

        print("getprogressapiCloud ", "running", req, "current", self.uuid)
        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        # re-calculate progress base on console progress
        total_steps = shared.total_tqdm.get_total_steps() 
        current_step = shared.total_tqdm.step 
        progress = 0.01 if (total_steps <= 0 or current_step <= 0) else current_step / total_steps

        progress = min(progress, 0.99)

        return models.ProgressResponse(progress=progress, uuid=self.uuid, eta_relative=eta_relative, 
                status="running", state=shared.state.dict(), textinfo=shared.state.textinfo)

    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        print("Processing extras_batch_images_api")

        reqDict = setUpscalers(req)
        reqDict['extras_upscaler_1'] = "R-ESRGAN 4x+" if reqDict["extras_upscaler_1"] == 'None' else reqDict["extras_upscaler_1"]
        reqDict['upscaling_resize'] = 2 if reqDict["upscaling_resize"] == 'None' else reqDict["upscaling_resize"]

        print("Args: " +  str(reqDict))
        save_dir_path = reqDict.pop('save_dir', None)
        send_paths = reqDict.pop('send_paths', True)
        reqDict.pop('image_count', None)
        reqDict.pop('run_times', None)
        image_list = reqDict.pop('imageList', [])
        image_folder = []
        total_steps = 0
        for x in image_list:
            image_data = None
            if os.path.exists(x.name) and not x.data:
                image_data = Image.open(os.path.abspath(x.name))
            elif x.data is not None:
                image_data = decode_base64_to_image(x.data)

            if image_data is not None:
                image_folder.append(image_data)
                total_steps += postprocessing.scripts.scripts_postproc.get_total_steps(image_data.width, image_data.height, reqDict['extras_upscaler_1'], reqDict['upscaling_resize'])

        def update_step_func(data):
            update_steps = postprocessing.scripts.scripts_postproc.update_step_num(data)
            if (update_steps > 0):
                for _ in range(update_steps):
                    shared.total_tqdm.update()

        print_capture = WriteProcessor(update_step_func)

        error_code = ErrorCode.OK.value
        devices.torch_gc()
        with self.queue_lock:
            shared.total_tqdm.updateTotal(total_steps)
            p = postprocessing
            def process_upscale(p):
                try:
                    with redirect_stdout(print_capture):
                        result = p.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)
                except Exception as ex:
                    raise ex
                return result
            result, error_code = self.generation_oom_process_handle("extras_batch_images_api", p,
                process_upscale, shared.cmd_opts.memory_ratio_allow, shared.cmd_opts.memory_timeout,
                shared.cmd_opts.memory_timeout, shared.cmd_opts.memory_check_interval)

            postprocessing.scripts.scripts_postproc.image_changed()
            postprocessing.scripts.scripts_postproc.release_model()

            devices.torch_gc()
            shared.total_tqdm.clear()

        b64images = []
        image_paths = []
        if (shared.state.interrupted == False and isinstance(result, tuple) and len(result) >= 1 and result[0] is not None):
            if (send_paths):
                for image in result[0]:
                    if (shared.state.interrupted):
                        break
                    try:
                        image = self.watermark.encode(image)
                        path, _ = images.save_image(image, save_dir_path, basename="")
                        image_paths.append(path)
                    except Exception as e:
                        if hasattr(e, 'errno'):
                            if e.errno == errno.ENOSPC:
                                print("Not enough disk space for saving images")
                                error_code = ErrorCode.ErrSpace.value
                                break
                            else:
                                print(str(e))
                                raise e
                        else:
                            print(str(e))
                            raise e
            else:
                for i in range(result[0]):
                    result[0][i] = self.watermark.encode(result[0][i])
                b64images = list(map(encode_pil_to_base64, result[0]))

        if (shared.state.interrupted):
            for path in image_paths:
                try:
                    os.remove(path)
                except Exception as ex:
                    print(f"Remove file {path} for interrupt error: {str(ex)}")
            image_paths = []
            b64images = []

        if (shared.state.interrupted == False and len(image_paths) == 0 and len(b64images) == 0 and error_code == ErrorCode.OK.value):
            error_code = ErrorCode.ErrData.value
        html_info = ""
        if (isinstance(result, tuple) and len(result) >= 2 and result[1] is not None):
            html_info = result[1]

        return models.ExtrasBatchImagesResponse(images=b64images, image_paths=image_paths, html_info=html_info, error=error_code)

    def pnginfoapi(self, req: models.PNGInfoRequest):
        if(not req.image.strip()):
            return models.PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{'parameters': geninfo}, **items}

        return models.PNGInfoResponse(info=geninfo, items=items)

    def progressapi(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def getprogressapi(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        # re-calculate progress base on console progress
        total_steps = shared.total_tqdm.get_total_steps() 
        current_step = shared.total_tqdm.step 
        progress = 0.01 if (total_steps <= 0 or current_step <= 0) else current_step / total_steps

        progress = min(progress, 0.99)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        return models.InterrogateResponse(caption=processed)

    def interruptapiCloud(self, data: dict):

        print("interruptapi", data)
        if data["uuid"] == self.uuid:
            shared.state.interrupt()
            return {"status": "done"}
        else:
            with self.write_lock:
                if data["uuid"] in self.uuid_lst:
                    del self.uuid_lst[self.uuid_lst.index(data["uuid"])]
                    return {"status": "failed"}
            return {"status": "undefined"}

        return {}
        
    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        checkpoint_name = req.get("sd_model_checkpoint", None)
        if checkpoint_name is not None and checkpoint_name not in checkpoint_aliases:
            raise RuntimeError(f"model {checkpoint_name!r} not found")

        for k, v in req.items():
            shared.opts.set(k, v, is_api=True)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_latent_upscale_modes(self):
        return [
            {
                "name": upscale_mode,
            }
            for upscale_mode in [*(shared.latent_upscale_modes or {})]
        ]

    def get_sd_models(self):
        import modules.sd_models as sd_models
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in sd_models.checkpoints_list.values()]

    def get_sd_vaes(self):
        import modules.sd_vae as sd_vae
        return [{"model_name": x, "filename": sd_vae.vae_dict[x]} for x in sd_vae.vae_dict.keys()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self, req: models.EmbeddingRequest = Depends()):
        db = sd_hijack.model_hijack.embedding_db
        if req.refresh_lora:
            print("refreshing loras")
            shared.refresh_loras_func()
        
        if req.refresh_embedding:
            print("refreshing embeddings")
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

        if req.select_model:
            print("Reload model list")
            list_models()
            print(f"Change model request: from {shared.opts.sd_model_checkpoint} to {req.select_model}")
            if (req.select_model not in checkpoints_list and os.path.exists(req.select_model)):
                model_path = req.select_model.lower()
                if (model_path in checkpoints_path_list):
                    req.select_model = checkpoints_path_list[model_path]
                else:
                    print(f"Could not add model {req.select_model} to list")

            if (req.select_model in checkpoints_list):
                shared.opts.sd_model_checkpoint = req.select_model
                reload_model_weights()
            else:
                print(f"Could not find model {req.select_model} in list")

        if req.select_model:
            print(f"Change model request: from {shared.opts.sd_model_checkpoint} to {req.select_model}")
            if (req.select_model not in checkpoints_list and os.path.exists(req.select_model)):
                print("Reload model list")
                list_models()
                model_path = req.select_model.lower()
                if (model_path in checkpoints_path_list):
                    req.select_model = checkpoints_path_list[model_path]
                else:
                    print(f"Could not add model {req.select_model} to list")

            if (req.select_model in checkpoints_list):
                shared.opts.sd_model_checkpoint = req.select_model
                reload_model_weights()
            else:
                print(f"Could not find model {req.select_model} in list")

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
                "alias": embedding.alias,
                "type": 0,
            }
        
        def convert_lora(lora):
            return {
                "alias": lora.config_alias,
                "type": 1,
            }
        
        def convert_model(id, path):
            from pathlib import Path
            alias = Path(id).stem
            ret = {
                "id": id,
                "alias": alias,
                "selected": False             
            }
            if (id == shared.opts.sd_model_checkpoint):
                ret["selected"] = True
            return ret

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}
        
        def convert_loras(loras):
            return {loraStyle.prompt: convert_lora(loraStyle) for loraStyle in loras.values()}
        
        def convert_models(check_point_list):
            return {v: convert_model(v, k) for (k,v) in check_point_list.items()}
        
        loadedStyles = convert_embeddings(db.word_embeddings)
        loadedLoras = convert_loras(shared.available_loras)
        allStyles = {}
        allStyles.update(loadedStyles)
        allStyles.update(loadedLoras)

        sampleMethods = {sampler[0]: {"name": sampler[0], "aliases": sampler[2], "options": sampler[3]} for sampler in sd_samplers.all_samplers}
        models = convert_models(checkpoints_path_list)

        device_id = -1
        if (torch.cuda.is_available()):
            device_id = torch.cuda.current_device()

        while self.initDone == False:
            time.sleep(0.2)

        print("api get_embeddings and samplers")
        # print(f"DeviceID: {device_id}")
        # print(f"Styles: {loadedStyles}")
        # print(f"Loras: {loadedLoras}")
        # print(f"Sampling Methods: {sampleMethods}")
        # print(f"Models: {models}")
        if not self.uuid_lst:
            self.estimetime = 0
            self.imgQueue = 0
        return {
            "device_id": device_id,
            "samplers": sampleMethods,
            "loaded": allStyles,
            "skipped": convert_embeddings(db.skipped_embeddings),
            "models": models,
        }

    def refresh_checkpoints(self):
        with self.queue_lock:
            shared.refresh_checkpoints()

    def refresh_vae(self):
        with self.queue_lock:
            shared_items.refresh_vae_list()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin(job="create_embedding")
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            return models.CreateResponse(info=f"create embedding filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create embedding error: {e}")
        finally:
            shared.state.end()


    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="create_hypernetwork")
            filename = create_hypernetwork(**args) # create empty embedding
            return models.CreateResponse(info=f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create hypernetwork error: {e}")
        finally:
            shared.state.end()

    def preprocess(self, args: dict):
        try:
            shared.state.begin(job="preprocess")
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return models.PreprocessResponse(info='preprocess complete')
        except KeyError as e:
            return models.PreprocessResponse(info=f"preprocess error: invalid token: {e}")
        except Exception as e:
            return models.PreprocessResponse(info=f"preprocess error: {e}")
        finally:
            shared.state.end()

    def train_embedding(self, args: dict):
        try:
            shared.state.begin(job="train_embedding")
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as msg:
            return models.TrainResponse(info=f"train embedding error: {msg}")
        finally:
            shared.state.end()

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="train_hypernetwork")
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as exc:
            return models.TrainResponse(info=f"train embedding error: {exc}")
        finally:
            shared.state.end()

    def get_memory(self):
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            ram = { 'error': f'{err}' }
        try:
            import torch
            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                cuda = {'error': 'unavailable'}
        except Exception as err:
            cuda = {'error': f'{err}'}
        return models.MemoryResponse(ram=ram, cuda=cuda)

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive, root_path=root_path)

    def kill_webui(self):
        restart.stop_program()

    def restart_webui(self):
        if restart.is_restartable():
            restart.restart_program()
        return Response(status_code=501)

    def stop_webui(request):
        shared.state.server_command = "stop"
        return Response("Stopping.")

