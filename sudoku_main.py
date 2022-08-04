# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import plotCVImg
import correction
import extractNumber
from PIL import ImageGrab
from config import *


while True:
    try:
        img = ImageGrab.grabclipboard().convert('RGB')
        # print(type(im))
        img_original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        break
    except AttributeError:
        print('剪贴板中没有图片! ')
        input()

# 存储题目的数组 shape=(9*9, 30*30)
sudoku = np.zeros(shape=(9 * 9, NUM_WIDTH * NUM_HEIGHT))

# 读取图片 read image
# img_original = cv2.imread('./images/c3.png')
if DEBUG:
    plotCVImg.plotImg(img_original, "original")

# 预处理及图像校正 pre-processing and image correction
img_puzzle = correction.correct2(img_original)
if DEBUG:
    plotCVImg.plotImg(img_puzzle, "pre-process")


# 识别并记录序号 detect numbers and extract them
indexes_numbers = []
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        img_number = img_puzzle[i * GRID_HEIGHT:(i + 1) * GRID_HEIGHT][:, j * GRID_WIDTH:(j + 1) * GRID_WIDTH]
        # plotCVImg.plotImg(img_number, f"{i}, {j}")
        hasNumber, sudoku[i * 9 + j, :] = extractNumber.extract_number(img_number)
        if hasNumber:
            indexes_numbers.append(i * 9 + j)

# 显示提取数字结果
if DEBUG:
    print("There are", len(indexes_numbers), "numbers and the indexes of them are:")
    print(indexes_numbers)
    # 创建子图
    rows = len(indexes_numbers) // 5 + 1
    f, axarr = plt.subplots(rows, 5)
    row = 0
    for x in range(len(indexes_numbers)):
        ind = indexes_numbers[x]
        if (x % 5) == 0 and x != 0:
            row = row + 1
        axarr[row, x % 5].imshow(cv2.resize(sudoku[ind, :].reshape(NUM_WIDTH, NUM_HEIGHT), (NUM_WIDTH * 5, NUM_HEIGHT * 5)), cmap=plt.gray())
    for i in range(rows):
        for j in range(5):
            axarr[i, j].axis("off")
    plt.show()

# 单个数字图
img = np.zeros(shape=(len(indexes_numbers), NUM_WIDTH, NUM_HEIGHT))
for num in range(len(indexes_numbers)):
    img[num] = sudoku[indexes_numbers[num]].reshape(NUM_WIDTH, NUM_HEIGHT)
digit_img = cv2.hconcat(img)
plotCVImg.plotImg(digit_img, "digit")

