import imageio.v3 as iio
import os
import glob
import tqdm

# species = 'papaya'
# species = 'geranium'
# species = 'betel'
# species = 'thevetia'
# species = 'ficus'
species = 'soybean'
# species = 'maize'
# species = 'roses'

folder = 'sample_leaf_data/{}'.format(species)  # Path to the folder containing images

files = glob.glob(os.path.join(folder, '*.jpg'))
files.sort()
for i, file in enumerate(tqdm.tqdm(files)):
     
     # continue if is digit in the filename
     if os.path.basename(file).split('.')[0].isdigit():
          continue

     img = iio.imread(file)

     os.rename(file, os.path.join(folder, f'{i:05d}.jpg'))
     # os.rename(file.replace('.jpg', '_keypoints.txt'), os.path.join(folder, f'{i:03d}.txt'))

     # map_dict[os.path.basename(file)] = f'{i:03d}.jpg'

     print(f'{file} -> {i:05d}.jpg')