## Hybrid Image
&emsp;&emsp;
  we use a high-pass filter and a low-pass filter to process two image and then merge them into one image. In close range, we mainly gain visual information from high-frequency signals. Instead, low-frequency signals provide more infomation in distance. Hence we can see two different pictures from different distance in one image.<br>
&emsp;&emsp;
  Some examples are given as follows.<br>
  ![ColoredResult](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result1Color.jpg)<br>
  ![ColorMagnitude](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result2Color.jpg)<br>
&emsp;&emsp;
  In this case, you can see a school badge in close and a building in distance.<br>
&emsp;&emsp;
  This algorithm is stiil vaild for grayscale. Due to exclusion of color, we even observe a better result<br>.
  ![GrayscaleResult](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result1Gray.jpg)<br>
  ![[GrayscaleMagnitude](https://github.com/NK-CS-ZZL/computer-vision/blob/master/hybridImage/figures/result2Gray.jpg)<br>
