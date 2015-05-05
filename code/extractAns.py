# -*- coding: utf-8 -*-
import sys
import os
import pickle
import numpy as np

f1 = open("ans.csv", "r")
f2 = open("test.csv", "r")

dict = {}

f1.readline()

while(True):
	row = f1.readline()
	if not row:
		break
	row = row.split(',')
	key = row[1] + " " + str(row[5])
	val = row[16].strip('\n')
	dict[key] = int(val)

f1.close()

f2.readline()
Y = []

while(True):
	row = f2.readline()
	if not row:
		break
	row = row.split(',')
	(date, time) = row[0].split(' ')
	hour = int(time.split(':')[0])
	key = date + " " + str(hour)
	if dict.has_key(key):
		Y.append(dict[key])

f2.close()
Y = np.array(Y)

np.save("answers.npy", Y)   #Saved the actual answers in the form of numpy array