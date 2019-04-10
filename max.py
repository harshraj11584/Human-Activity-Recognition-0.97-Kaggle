import numpy as np
import pandas as pd
from collections import Counter

p1=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.03896103896103.txt')
#print(p1.head(8))
#print(p1.shape)
p2=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.03896103896103(1).txt')
#print(p2.head(8))
p3=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.03896103896103(2).txt')
#print(p3.head(8))
p4=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.14285714285714.txt')
#print(p4.head(8))
p5=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.19480519480518.txt')
#print(p5.head(8))
p6=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.24675324675324.txt')
#print(p6.head(8))
p7=pd.read_csv('/home/harsh/Desktop/Human Activity Recognition/kernel/p97.71428571428571.txt')
print("Id,Predicted Label")
for i in range(1,6419):
	curr_label=[]
	curr_label.append(p1["Predicted Label"][i-1])
	curr_label.append(p2["Predicted Label"][i-1])
	curr_label.append(p3["Predicted Label"][i-1])
	curr_label.append(p4["Predicted Label"][i-1])
	curr_label.append(p5["Predicted Label"][i-1])
	curr_label.append(p6["Predicted Label"][i-1])
	curr_label.append(p7['Predicted Label'][i-1])
	
	#print(curr_label)	
	most_common,num_most_common = Counter(curr_label).most_common(1)[0]
	print(str(i-1)+","+str(most_common))
	#print(num_most_common)

