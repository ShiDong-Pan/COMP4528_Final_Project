### This is a project source code explanation of team 6 in 2019 ENGN4528.
#### Please run UI.py in terminal to test the program.
##### Here is a sample commond to run UI.py
```python
python3 UI.py -t sample_image/greek_gray.jpg -c sample_image/greek_color.jpg
```
where -t indicates the path of test image (grayscale image) and -c indicates the path of any color image. <br/>Please use our UI to get color from color image and hint any color in the test image. 
1. Select color from right image (color image) by clicking any pixel when the status is 0. 
2. Then press the button line to make status to 1 and hint color on the left image (gray image). 
3. If you want to select other colors, just make status to 0 agian and repeat above steps.
4. If you finish the hinting, just click "q" in the keyboard. After a while, you will see the colorization result!
* If you already have grayscale image with corresponding hinted image, you can set the path of these two images and directly run Colorization.py, and you will see you result after a while.