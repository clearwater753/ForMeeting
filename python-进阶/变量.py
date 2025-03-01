print(ord('?'))
print(int(0x3f))
print('???')

height = 1.75
weight = 80.5

bmi = height/pow(weight,2)

if bmi < 18.5:
    print("过轻了")
else:
    print("没有过轻")

d = dict()

for i in range(10):
    d[i] = i
print(type(list(d.keys())[0]))
s = set()
l = []
t = (1, 2, 3)
# t2 = (1, 2, [3, 4])
s.add(t)
# s.add(t2)
print(s)