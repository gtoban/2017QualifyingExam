import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

x = np.arange(1000)
print(x[0])
plt.figure()
fig1 = plt.plot(x)
plt.savefig("xrange1.png")

plt.figure()
fig2 = plt.plot(-x)
plt.savefig("xrange2.png")

plt.figure()
fig2 = plt.plot(-x)
fig1 = plt.plot(x)
plt.savefig("xrange.png")

file = open("test.txt", "w")
file.write("test\ntest")
file.write("lineTest")
file.close()

#plt.figure(
#legend((fig1,fig2),('postive', 'negative'))

