import math
import random
import matplotlib.pyplot as plt
import itertools
from io import BytesIO
from PIL import Image as Image  # pillow
from typing import Iterable
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def getInfomationEntropy(str_: str):
    strLen = len(str_)
    set_ = set(str_)
    infomationEntropy = 0
    for i in set_:
        charCount = str_.count(i)
        entroptForChar = -(charCount / strLen) * math.log(charCount / strLen, 2)
        infomationEntropy += entroptForChar
    return infomationEntropy


def img2gif(imgs: Iterable, output: Path, **kwargs):
    """
    图像转gif
    :param imgs: 图片
    :param output: gif输出(str | Path | file object)
    """
    kwargs.setdefault("save_all", True)
    kwargs.setdefault("loop", True)
    imgs = [Image.open(img) for img in imgs]

    imgs[0].save(output, append_images=imgs[1:], **kwargs)


# while 1:
#     s = input()
#     s=s.replace(' ','')
#     print('='*50+'\n'+s+'\n'+'='*50)
#     sl = len(s)
#     csl = len(set(s))
#     a = getInfomationEntropy(s)
#     print('字符串长度——%d\n信息熵(单个字符)——%f\n编码极限(bit)——%f\n\n单字符信息量%f/字符集大小%d=字符利用率%f' %
#           (sl, a, a*sl, 2**a, csl, 2**a/csl))


x = []
y = []

fullChoiceSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890!@#$%^&*()丰王井开夫天无元专云扎艺木五支厅不太犬区历尤友屯比互切少日中冈贝内水见午牛手仁什片仆化仇币仍仅斤爪反介父从今凶分公仓月氏勿欠风丹匀乌凤勾文六方火为户认心尺引丑巴孔队办以双书幻"
fullChoiceSet_len = len(fullChoiceSet)
imgs = []
for choices_range in range(1, fullChoiceSet_len + 1):
    choiceSet = fullChoiceSet[:choices_range]
    x.clear()
    y.clear()
    for _, i in itertools.product(range(100), range(128)):
        str_ = str().join(random.choices(choiceSet, k=i))
        # str_ = input()
        infomationEntropy = getInfomationEntropy(str_)
        x.append(i)
        y.append(infomationEntropy)
    print("%d|%d" % (choices_range, fullChoiceSet_len))
    new_img = BytesIO()
    imgs.append(new_img)
    plt.cla()
    plt.scatter(x, y, alpha=0.3, s=2)
    plt.xlim(0, 128)
    plt.ylim(0, math.log2(fullChoiceSet_len) + 1)
    plt.xlabel("随机字符串长度")
    plt.ylabel("信息熵")
    plt.title("随机字符串的字符集大小: %d" % choices_range)
    plt.savefig(new_img, dpi=100)

# imgs = Path("plt").iterdir()
# imgs = list(imgs)
# imgs.sort(key=lambda x:int(str(x.stem)))
img2gif(imgs, output="InfomationEntropy.gif")
