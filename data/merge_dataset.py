import os

def get_file_paths(folder_path='.'):
  txt_files = []
  for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
      txt_files.append(file_name)
  return txt_files

files = get_file_paths()
print(files)

dataset = ""


for file in files:
  with open(file, 'r') as f:
    text = f.read()
  dataset += text
  dataset += "\n\n"

if not os.path.exists('dataset.txt'):
  with open('dataset.txt', 'w') as f:
    f.write(dataset)