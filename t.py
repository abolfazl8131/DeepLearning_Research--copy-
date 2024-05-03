import json
f = open('ssss' , 'r')
d = json.loads(f.read())

d['name'] = 'abol'
d['age'] = 22
print(d)