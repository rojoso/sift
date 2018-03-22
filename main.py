import sift
from numpy import *
from pylab import *
from PIL import Image
import harris
imname1 = 'timg1.jpg'
imname2 = 'timg2.jpg'

im1 = array(Image.open(imname1).convert('L'))
sift.process_image(imname1,'ti1.sift')

l1,d1 = sift.read_features_file('./ti1.sift')

im2 = array(Image.open(imname2).convert('L'))
sift.process_image(imname2,'ti2.sift')

l2,d2 = sift.read_features_file('./ti2.sift')

locs1 = l1[:,:2]
locs2 = l2[:,:2]

matchscore = sift.match_twoside(d1,d2)
print(locs1.shape)
print(locs2.shape)
print(matchscore.shape)
print(matchscore.max())
print(locs1[24,1])

#figure()
#sift.plot_feature(im1,l1,circle = True)

#show()
figure()
harris.plot_matches(im1,im2,locs1,locs2,matchscore)
show()


