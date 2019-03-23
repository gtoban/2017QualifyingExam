import numpy as np
import pandas as pd
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


test = pd.read_csv("w2t.csv", header=None)
tn = np.array(test).T
tn.shape = (1131,)
new = []
w1f = open("w2t2.csv","w")
for i in range(1131):
    #print(tn[i])
    temp = tn[i][1:-1].split(' ')
    row = []
    for item in temp:
        if is_number(item):
    #        print("item:",item)
            row.append(item)
    #    else:
    #        print("bad:",item)
    if len(row) > 2:
        print("BAD ROW", i)
    w1f.write(row[0]+","+row[1]+"\n")
        
w1f.close()

