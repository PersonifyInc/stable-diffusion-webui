import numpy as np
import cv2
import math
import os
from PIL import Image

wmResize = 4
wmEmbedSize = 16
class EmbedDwtDctSvd(object):
    def __init__(self, block=4, waterMPath="watermark.png"):
        self._block = block
        self.w0 = 0.5
        self.w1 = -0.5
        self.s0 = 0.5
        self.s1 = 0.5
        self._watermarkDiff = None
        self._watermarkPixels = None
        self._watermarkCache = {}
        self.createWM(waterMPath)

    def decode(self, data):
        if isinstance(data, str) and os.path.isfile(data):
            # data = cv2.imread(data)
            data = Image.open(data)

        if isinstance(data, Image.Image):
            data = np.array(data)#[...,::-1]

        bgr = self.readImgDecode(data)
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        u = self.RgbToU(bgr[..., 0], bgr[..., 1], bgr[..., 2])
        ca1 = self.FWT(u[:row//4*4,:col//4*4], 2)
        ca1 = ca1[:ca1.shape[0]//4, :ca1.shape[1]//4]
        wmI, sim = self.decode_frame(ca1)

        # cv2.imwrite("wmOut.png", wmI)
        return wmI, sim

    def createWM(self, waterMPath):
        self._watermarkPixels = cv2.imread(waterMPath)
        temp = np.zeros((self._watermarkPixels.shape[1], self._watermarkPixels.shape[0], 3))
        for x in range(self._watermarkPixels.shape[1]):
            for y in range(self._watermarkPixels.shape[0]):
                temp[x, y] = self._watermarkPixels[y, x]
        self._watermarkPixels = temp
        self._watermarkDiff = self.GenerateWatermarkDiff(512, 512)

    def encode(self, data):
        if isinstance(data, str) and os.path.isfile(data):
            # data = cv2.imread(data)
            data = Image.open(data)
        if isinstance(data, Image.Image):
            data = np.array(data)#[...,::-1]

        bgr = data
        if self._watermarkPixels is None:
            self.createWM()
        if self._watermarkDiff is None:
            self._watermarkDiff = self.GenerateWatermarkDiff(512, 512)

        (height, width, channels) = bgr.shape
        if f'{height}x{width}' not in self._watermarkCache:
            self._watermarkCache[f'{height}x{width}'] = cv2.resize(self._watermarkDiff, (width//wmResize, height//wmResize))[..., ::-1]
        img = self.MergeWatermarkPixels(bgr)

        # cv2.imwrite("deOut.png", img)
        img = Image.fromarray(np.uint8(img)) #[...,::-1]))

        return img

    def RgbData(self, gray):
        (height, width) = gray.shape
        rgb = np.zeros((height, width, 3))
        rgb[..., 0] = gray
        rgb[..., 1] = gray
        rgb[..., 2] = gray

        return rgb

    def ToYuv(self, rgb):
        (height, width, channels) = rgb.shape
        yuv = np.zeros((height, width, 3))

        yuv[..., 0] = self.RgbToY(rgb[..., 0], rgb[..., 1], rgb[..., 2])
        yuv[..., 1] = self.RgbToU(rgb[..., 0], rgb[..., 1], rgb[..., 2])
        yuv[..., 2] = self.RgbToV(rgb[..., 0], rgb[..., 1], rgb[..., 2])

        return yuv

    def ToRgb(self, Y, U, V):
        (height, width) = Y.shape
        rgb = np.zeros((height, width, 3))

        rgb[..., 0] = self.YuvToR(Y, U, V);
        rgb[..., 1] = self.YuvToG(Y, U, V);
        rgb[..., 2] = self.YuvToB(Y, U, V);

        return rgb

    def ToByte(self, value):
        if value < 0:
            return 0
        if value > 255:
            return 255
        return value

    def MergeWatermarkPixels(self, image):
        from joblib import Parallel, delayed
        (height, width, channels) = image.shape
        
        count = 0
        pixelSize = 3

        imagePx = image
        waterPx = self._watermarkCache[f'{height}x{width}']
        (height_2, width_2, _) = self._watermarkCache[f'{height}x{width}'].shape
        def threadMerge(h, imgH):
            for w in range(0, width_2):
                nextSame = False
                prevSame = False
                wP = (width - width_2) + w
                if (wP) > 0 and imgH[wP, 0] == imgH[wP-1, 0] and imgH[wP, 1] == imgH[wP-1, 1]:
                    nextSame = True

                if (wP+1) < width and imgH[wP, 0] == imgH[wP+1, 0] and imgH[wP, 1] == imgH[wP+1, 1]:
                    prevSame = True

                if not nextSame or not prevSame:
                    # imagePx[i] = self.ToByte(imagePx[i] + 128 - waterPx[i])
                    # imagePx[i + 1] = self.ToByte(imagePx[i + 1] + 128 - waterPx[i + 1])
                    imgH[wP, 0] = min(255, max(0, imgH[wP, 0] + 128 - waterPx[h, w, 0]))
                    imgH[wP, 1] = min(255, max(0, imgH[wP, 1] + 128 - waterPx[h, w, 1]))
            return imgH
        result = Parallel(n_jobs=4)(delayed(threadMerge)(h, imagePx[(height-height_2) + h]) for h in range(height_2))
        imagePx[height-height_2:] = np.array(result)

        # imagePx = np.reshape(image, -1)
        # waterPx = np.reshape(self._watermarkCache[f'{height}x{width}'], -1)
        # for h in range(height):
            # hPos = h * width * pixelSize
            # for w in range(0, width*pixelSize, pixelSize):
            #     count += 1
            #     i = hPos + w

            #     nextSame = False
            #     prevSame = False
            #     if (i) > 0 and imagePx[i] == imagePx[i-pixelSize] and imagePx[i + 1] == imagePx[i + 1 - pixelSize]:
            #         nextSame = True

            #     if (i + pixelSize) < len(imagePx) and imagePx[i] == imagePx[i + pixelSize] and imagePx[i + 1] == imagePx[i + 1 + pixelSize]:
            #         prevSame = True

            #     if not nextSame or not prevSame:
            #         # imagePx[i] = self.ToByte(imagePx[i] + 128 - waterPx[i])
            #         # imagePx[i + 1] = self.ToByte(imagePx[i + 1] + 128 - waterPx[i + 1])
            #         imagePx[i] = max(255, min(0, imagePx[i] + 128 - waterPx[i]))
            #         imagePx[i + 1] = max(255, min(0, imagePx[i + 1] + 128 - waterPx[i + 1]))
        outImage = np.reshape(imagePx, (height, width, channels))

        return outImage

    def GenerateWatermarkDiff(self, diffW, diffH):
        gray = np.zeros((diffH, diffW))

        rgbData = self.RgbData(gray)
        yuv = self.ToYuv(rgbData)

        yuv[..., 1] = self.EmbedWatermarkInit(yuv[..., 1])
        rgb = self.ToRgb(yuv[..., 0], yuv[..., 1], yuv[..., 2])

        rgb = 128 - rgb

        return rgb


    def EmbedWatermarkInit(self, data):
        BlockSize = 4
        watermarkData = np.zeros((self._watermarkPixels.shape[0], self._watermarkPixels.shape[1]))
        watermarkData[self._watermarkPixels[..., 0] > 125] = 255


        watermarkData2 = np.zeros((self._watermarkPixels.shape[0], self._watermarkPixels.shape[1]))
        for x in range(self._watermarkPixels.shape[0]):
            for y in range(self._watermarkPixels.shape[1]):
                watermarkData2[x, y] = 255 if self._watermarkPixels[x, y, 0] > 125 else 0

        data = self.FWT(data, 2)
        subband = data[:data.shape[0]//4, :data.shape[1]//4]

        for y in range(self._watermarkPixels.shape[0]):
            for x in range(self._watermarkPixels.shape[1]):
                block = subband[x * BlockSize : x * BlockSize + BlockSize, y * BlockSize : y * BlockSize + BlockSize]

                block = cv2.dct(block)
                midbandSum = max(2, abs(self.MidBand(block)))
                # sigm = (watermarkData[x, y] > 125 ? 3 : -3)
                sig = 1 if watermarkData[x, y] > 125 else -1

                block[1, 2] += midbandSum * sig
                block[2, 0] += midbandSum * sig
                block[2, 1] += midbandSum * sig
                block[2, 2] += midbandSum * sig

                block = cv2.idct(block)
                subband[x * BlockSize : x * BlockSize + BlockSize, y * BlockSize : y * BlockSize + BlockSize] = block

        data[:data.shape[0]//4, :data.shape[1]//4] = subband
        data = self.IWT(data, 2)
        return data


    def FWTSingle(self, data):
        h = len(data) >> 1
        temp = np.zeros(len(data))
        for i in range(h):
            k = i << 1
            temp[i] = data[k] * self.s0 + data[k+1] * self.s1
            temp[i+h] = data[k] * self.w0 + data[k+1] * self.w1

        data = temp
        return data

    def IWTSingle(self, data):
        h = len(data) >> 1
        temp = np.zeros(len(data))
        for i in range(h):
            k = i << 1
            temp[k] = (data[i] * self.s0 + data[i+h] * self.w0) / self.w0
            temp[k+1] = (data[i] * self.s1 + data[i+h] * self.w1) / self.s0

        data = temp
        return data

    def FWT(self, data, itera):
        (rows, cols) = data.shape

        for k in range(itera):
            lev = 1 << k;
            levCols = int(cols /lev)
            levRows = int(rows / lev)
            for i in range(levRows):
                data[i] = self.FWTSingle(data[i])

            for j in range(levCols):
                data[:, j] = self.FWTSingle(data[:, j])
        return data

    def IWT(self, data, itera):
        (rows, cols) = data.shape

        for k in range(itera-1, -1, -1):
            lev = 1 << k;
            levCols = int(cols /lev)
            levRows = int(rows / lev)

            for j in range(levCols):
                data[:, j] = self.IWTSingle(data[:, j])

            for i in range(levRows):
                data[i] = self.IWTSingle(data[i])

        return data

    def RgbToY(self, red, green, blue):
        return 0.299 * red + 0.587 * green + 0.114 * blue

    def RgbToU(self, red, green, blue):
        return -0.147 * red - 0.289 * green + 0.436 * blue

    def RgbToV(self, red, green, blue):
        return 0.615 * red - 0.515 * green - 0.100 * blue

    def YuvToR(self, y, u, v):
        return y + 1.140 * v

    def YuvToG(self, y, u, v):
        return y - 0.395 * u - 0.581 * v

    def YuvToB(self, y, u, v):
        return y + 2.032 * u

    def decode_frame(self, frame):
        (row, col) = frame.shape
        num = 0

        gray = np.zeros((32, 32))
        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block : i*self._block + self._block,
                              j*self._block : j*self._block + self._block]

                gray[i, j] = self.infer_dct_svd(block)
                num = num + 1

        similarity = 0
        total = int((row//self._block) * (col//self._block))
        for i in range(row//self._block):
            for j in range(col//self._block):
                oldValue = 255 if (self._watermarkPixels[i, j, 0] > 125) else 0
                if gray[i, j] == oldValue:
                    similarity += 1

        similarity = int((similarity / total) * 100)
        return gray, similarity

    def MidBand(self, data):
        return np.sum([data[1, 2], data[2, 0], data[2, 1], data[2, 2]])

    def infer_dct_svd(self, block):
        block = cv2.dct(block)
        score = 255 if (self.MidBand(block) > 0) else 0
        return score

    def readImgDecode(self, bgr):
        # bgr = cv2.imread(data)
        bgr = bgr[-bgr.shape[0]//wmResize:, -bgr.shape[1]//wmResize:]
        # if bgr.shape[0] != 512 or bgr.shape[1] != 512:
        bgr = cv2.resize(bgr, (512, 512))
        bgr[:,:,2] = bgr[:,:,0]
        bgr[:,:,0] = 255

        return bgr


if __name__ == "__main__":
    import time
    encry = EmbedDwtDctSvd()
    start = time.time()
    encry.encode("00000-2924880136.png")
    start2 = time.time()
    wmI, sim = encry.decode("D:\\Workspace\\GitHub\\test\\dog_3.png")
    print(sim, time.time() -start2, time.time() -start)
    start = time.time()
    # for i in range(10):
    #     encry.encode("00000-2924880136.png", "genI.png")
    print(time.time() -start)

