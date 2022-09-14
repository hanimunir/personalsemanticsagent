import numpy as np
import matplotlib.pyplot as plt
import math
list1 = [0.273871,0.584362,0.726122,0.851190,0.841066,0.851190]
list2 = [0.380314,0.767422,0.819998,0.914022,0.919367,0.914022]
list3=[0.223834,0.587153,0.723436,0.849489,0.846240,0.849489]
AD_Recall=[0.41409,0.10360,0.03815,0.16322,0.1531,0.16322]
AD_Precsion=[0.4055,0.18435,0.0341404,0.12816,0.13350,0.12816]
AD_F1=[0.456106,0.0927871,0.043496,0.16955,0.166291,0.169544]
plt.ylabel("Recall, Precision, F1 score")
plt.xlabel("Number of iteration")
y= [0.2,0.3,0.5,0.6,0.7,0.8,0.9]
x = [0,50,100,150,200,250,300]
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
n = len(list1)
ddof=0
mean = sum(list1) / n
std_R=0.0272
std_P=0.028
std_F1=0.028
std_p_R=normal_dist(list1,mean,std_R)
print("this is")
print(std_p_R)
plt.plot(list1)
plt.plot(list2)
plt.plot(list3)

#setting the xticks. Note x1 and x2 are tuples, thus + is concatenation
n = len(list3)
ddof=0
mean = sum(list3) / n
print("standard deviation precsion 0.0272 Recall 0.028 F1 0.028")

d=[]
for x in list3:
	
	d=abs(x-mean)
	print(d)
sum=(d)/n
print("average")
print(sum)
plt.plot(d)

plt.show()