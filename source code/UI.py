import numpy as np
import matplotlib.pyplot as plt
from Colorization import colorization
import sys
import argparse
import cv2


def parse_command_line_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test_image",dest = "test_image",
                    metavar = "TEST_IMAGE",default = "test_gray.png")
    parser.add_argument("-c","--color_image",dest = "color_image",
                    metavar = "COLOR_IMAGE",default = "test_color.png")
    parser.add_argument("-m","--hint_image",dest = "hint_image",
                    metavar = "HINT_IMAGE",default = "test_marked.png")
    
    args = parser.parse_args()

    return args

def main():
    
    args = parse_command_line_arguments()

    test_gray = args.test_image
    test_color = args.color_image

    test_hint = painting(test_gray,test_color)
    colorization(test_gray,test_hint,1)


def painting(img,img_c):

    args = parse_command_line_arguments()
    img = cv2.imread(img)
    img_c = cv2.imread(img_c)
    img_c = cv2.resize(img_c,(img.shape[1],img.shape[0]))
    test_hint = args.hint_image

    global xg,yg,color,c_r,c_g,c_b
    global x_shape,y_shape
    c_r,c_g,c_b = 255,255,255

    global img_stack
    img_stack = np.hstack((img,img_c))
    x_shape = img_stack.shape[0]
    y_shape = img_stack.shape[1]
    img_stack_show = np.zeros((img_stack.shape[0],img_stack.shape[1],3))+255
    img_stack_show[:img_stack.shape[0], :img_stack.shape[1]] = img_stack


    img_stack = np.vstack((img_stack,img_stack[:10,:]))
    img_stack[x_shape:,:]=255,255,255


    cv2.namedWindow('image')

    switch = '0:OFF\n1:ON'
    cv2.createTrackbar(switch, 'image', 0, 1, callback)

    tag = True
    while(tag):
        
        cv2.imshow('image', img_stack)
        if cv2.getTrackbarPos(switch, 'image') == 1:
            cv2.setMouseCallback('image', Mouseback1)
        else:
            cv2.setMouseCallback('image', Mouseback2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            x = img.shape[0]
            y= img.shape[1]
            img_mark = img_stack[:x,:y]
            cv2.imwrite(test_hint,img_mark)
            break    

    cv2.destroyAllWindows()
    return test_hint


def callback(object):
    pass


def save_XY(x,y):
    global xg,yg
    xg=y
    yg=x


def Mouseback1(event, x, y, flags, param):
    if flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
        global xg,yg,color,img_stack,c_r,c_g,c_b
        cv2.circle(img_stack, (x, y), 4, [int(c_r),int(c_g),int(c_b)], 4)
        

def Mouseback2(event, x, y, flags, param):
    if flags == cv2.EVENT_FLAG_LBUTTON and not event == cv2.EVENT_MOUSEMOVE:
        global xg,yg,color, img_stack,c_r,c_g,c_b,x_shape
        save_XY(x,y)
        c_r,c_g,c_b = img_stack[xg,yg]
        img_stack[x_shape:,:]= int(c_r),int(c_g),int(c_b)
        print(xg,yg,c_r,c_g,c_b)


if __name__ == "__main__":
    main()