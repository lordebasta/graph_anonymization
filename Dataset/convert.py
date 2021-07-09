with open("simpsons.csv") as f:
    data = list(map(lambda l: l.rstrip().split()[:2], f.readlines()[1:]))

data = [','.join(line) + '\n' for line in data]
print(data)

with open("simposons.csv", "w") as of: 
    of.writelines(data)

