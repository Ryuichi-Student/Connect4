import os, re

rootdir = './dist'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        file = os.path.join(subdir, file)
        if file[:-3] == ".js":
            with open(file, 'r') as f: # open in readonly mode
                data = f.read()
            with open(file, 'w') as f: # open in readonly mode
              f.write(
                re.sub('.ts"', '.js"', data)
              )