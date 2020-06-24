import matplotlib.pyplot as plt
import imageio as im
import cv2
import matplotlib.gridspec as mg
from colorized_functions import *

def main():

    gray_image = 'sample_image/gray_1.jpg'
    hint_image = 'test_hint.png'
    wd_width = 1
    colorization(gray_image,hint_image,wd_width)

def colorization(gray_image,hint_image,wd_width):


    gray_img = im.imread(gray_image)
    try:
        gray_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
    except:
        pass
    gray_img = gray_img.astype(float)/255

    hint_img = im.imread(hint_image)
    hint_img = hint_img.astype(float)/255

    (img_rows, img_cols, _) = gray_img.shape

    img_size = img_rows * img_cols

    channel_Y,channel_U,channel_V = get_combined_yuv(gray_img,hint_img)

    map_colored = (abs(channel_U) + abs(channel_V)) > 0.0001

    img_yuv = np.dstack((channel_Y, channel_U, channel_V)) # combined image

    weightData = get_weightData(map_colored,img_yuv,wd_width,affinity_a) # weighting function

    mat = get_mat(weightData,img_rows,img_cols)

    u,v = get_uv(map_colored,img_yuv,img_rows,img_cols)


    colored_y = img_yuv[:,:,0].reshape(img_size, order='F')
    colored_u = spsolve(mat, u) # solve linear equations
    colored_v = spsolve(mat, v)
    colorized = yuv2rgb(img_yuv,colored_y,colored_u,colored_v)

    gs = mg.GridSpec(2,3)
    plt.figure(figsize=(20,10))
    plt.figure(1)
    plt.subplot(gs[0,0])
    plt.title('Gray_image')
    plt.imshow(gray_img)
    plt.subplot(gs[1,0])
    plt.title('Hint_image')
    plt.imshow(hint_img)
    plt.subplot(gs[:,1:])
    plt.title('Colorirzed_image')
    plt.imshow(colorized)

    plt.show()
    return colorized

if __name__ == "__main__":
    main()