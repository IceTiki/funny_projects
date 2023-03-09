from loguru import logger
import base64
import PIL.Image as PIL_Image
import numpy as np
from io import BytesIO
from pathlib import Path
import time
import requests
from matplotlib import pyplot as plt
from tikilib import math as tmath
from tikilib import text as ttext




@logger.catch
def convolution_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    二维卷积(边缘补0)
    """
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_canvas = np.zeros((img_h + 2, img_w + 2))
    img_canvas[1:-1, 1:-1] = img
    result_canvas = np.zeros([img_h, img_w])
    for i in range(img_h):
        for j in range(img_w):
            temp = img_canvas[i : i + 3, j : j + 3]
            temp = np.multiply(temp, kernel)
            result_canvas[i][j] = temp.sum()
    return result_canvas

def solve_slide(slide_img: str, canvas_img: str) -> dict:
    """
    滑块验证码解析
    :param slide_img, canvas_img: base64编码的图片
    :return dict: {
        "slide": 滑行像素,
        "canvas": 背景总长,
    }
    """
    # base64解码
    slide_img = base64.b64decode(slide_img)
    canvas_img = base64.b64decode(canvas_img)
    # 转为PIL解析图片
    slide_img = PIL_Image.open(BytesIO(slide_img)).convert("L")
    canvas_img = PIL_Image.open(BytesIO(canvas_img)).convert("L")
    # 转为ndarray
    slide_img: np.ndarray = np.array(slide_img)
    canvas_img: np.ndarray = np.array(canvas_img)

    # 找到滑块的有内容区域并裁剪
    img_range = np.nonzero(slide_img.sum(axis=1))[0]
    img_range = (np.min(img_range) - 5, np.max(img_range) + 5)
    slide_img, canvas_img = (
        slide_img[img_range[0] : img_range[1], :],
        canvas_img[img_range[0] : img_range[1], :],
    )

    # 卷积找边缘
    def find_edge(img: np.ndarray) -> np.ndarray:
        """
        利用sobel算子进行卷积, 查找图片边缘。返回值归一到0-255。
        :param img: np.ndarray
        """
        # sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # 计算x方向卷积
        img_x = convolution_2d(img, sobel_x)
        # 计算y方向卷积
        img_y = convolution_2d(img, sobel_y)
        # 得到梯度矩阵
        img_xy = np.sqrt(img_x**2 + img_y**2)
        # 梯度矩阵归一到0-255
        img_xy = img_xy * (255 / img_xy.max())
        return img_xy

    slide_xy = find_edge(slide_img)
    canvas_xy = find_edge(canvas_img)

    XE = []
    YE = []
    for x in range(canvas_xy.shape[1] - slide_xy.shape[1]):
        canvas_slide = np.zeros(canvas_xy.shape)
        canvas_slide[:, x : x + slide_xy.shape[1]] = slide_xy
        canvas_overlay = np.abs(canvas_slide - canvas_xy)
        XE.append(x)
        YE.append(np.sum(canvas_overlay))

    # matplotlib 绘图

    # output = Path("tmp_ourput")
    # output.mkdir(exist_ok=True)
    # plt.imshow(canvas_xy, cmap="gray_r")
    # plt.plot(XE, YE2)
    # plt.plot([np.argmin(YE2), np.argmin(YE2)], [0, 100])
    # plt.savefig(output / f"{time.time()}.png", bbox_inches="tight")
    # plt.cla()

    x = 0
    images = []
    duration = []
    while x < canvas_xy.shape[1] - slide_xy.shape[1]:
        canvas_slide = np.zeros(canvas_xy.shape)
        canvas_slide[:, x : x + slide_xy.shape[1]] = slide_xy
        canvas_overlay = np.abs(canvas_slide - canvas_xy)
        YE2 = tmath.normalization(YE[0 : x + 1]) * 100
        plt.imshow(canvas_overlay, cmap="gray_r")  # todo_tmp
        plt.plot(XE[0:x], YE2[0:x])
        plt.plot([x, x], [0, 100])
        img = BytesIO()
        plt.savefig(img, bbox_inches="tight")
        images.append(img)
        # plt.savefig(output / f"{x+10000}.png", bbox_inches="tight")
        plt.cla()
        dura = 1000 if tmath.normalization(YE)[x] < 0.3 else 20
        print(x, dura)
        duration.append(dura)
        x += 1
        # x += 10

    images = [PIL_Image.open(img) for img in images]
    images[0].save(
        "functions.gif",
        save_all=True,
        loop=True,
        append_images=images[1:],
        duration=duration,
    )

    return {
        "slide": np.argmin(YE),
        "canvas": canvas_img.shape[1],
    }


@logger.catch
def main():

    data = requests.get(
        r"https://authserver.gsau.edu.cn/authserver/common/openSliderCaptcha.htl"
    ).json()

    st = time.time()
    ttext.JsonFile.write(data)
    solution = solve_slide(data["smallImage"], data["bigImage"])
    logger.info(f"解——{solution['slide'] / solution['canvas'] * 280}")
    logger.info("时间——{}", time.time() - st)


if __name__ == "__main__":
    main()
