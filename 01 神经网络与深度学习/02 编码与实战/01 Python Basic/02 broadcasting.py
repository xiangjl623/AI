import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
print(A)
print(A.shape)  #(3,4)三行四列
print(A[2])
#sum的参数axis=0表示求和运算按列执行
#axis用来指明将要进行的运算是沿着哪个轴执行，在
#numpy中，0轴是垂直的，也就是列，而1轴是水平的，也就是行。
cal = A.sum(axis=0)
print(cal)
#百分比
percent = 100*A/cal.reshape(1,4)
print(percent)