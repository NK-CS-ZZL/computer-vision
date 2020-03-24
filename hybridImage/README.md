## Hybrid Image
&emsp;&emsp;
  we use a high-pass filter and a low-pass filter to process two image and then merge them into one image. In close range, we mainly gain visual information from high-frequency signals. Instead, low-frequency signals provide more infomation in distance. Hence we can see two different pictures from different distance in one image.<br>
&emsp;&emsp;
  Some examples are given as follows which contains both result images and magnitude visualizations in frequency domain<br>
  ![ColoredResult](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result1Color.jpg)<br>
  ![ColorMagnitude](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result2Color.jpg)<br>
&emsp;&emsp;
  In this case, you see a school badge in close and a building in distance.<br>
&emsp;&emsp;
  This algorithm is stiil vaild for grayscale. Due to exclusion of color, we even observe a better result.<br>.
  ![GrayscaleResult](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result1Gray.jpg)<br>
  ![GrayscaleMagnitude](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result2Gray.jpg)<br>
&emsp;&emsp;
  If you want to know more theory, please download [this](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/hybridImages.pdf).<br>
&emsp;&emsp;
  You should install opencv and a C++ compiler(or a IDE supporting C++) first to run these code. And my running environment lists below.<br>
#### Environment
- OS: Windows 10
- Opencv: [ver 4.2.0](https://sourceforge.net/projects/opencvlibrary/files/4.2.0/opencv-4.2.0-vc14_vc15.exe/download) (this link is for windows. If you use linux, you can find linux version in [official website](https://opencv.org/)).
- IDE: visual studio 2019
