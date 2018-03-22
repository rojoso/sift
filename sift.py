from PIL import Image
from numpy import *
from pylab import *
import os


def process_image(imgname,resultname,params = "--edge-thresh 5 --peak-thresh 5"):
	''' 将原始的图像处理称为sift文件格式'''
	if imgname[-3:] != 'pgm':
		im = Image.open(imgname).convert('L')
		#im = im.resize((200,250),Image.BILINEAR)
		im.save('temp.pgm')
		imgname = 'temp.pgm'
	cmmd = str("sift "+imgname + " --output=" + resultname + " "+ params)
	os.system(cmmd)
	print('proceeding',imgname,'to',resultname)

def read_features_file(filename):
	''' 读取特征属性值，然后让其以矩阵的形式返回'''

	f = loadtxt(filename)

	return f[:,:4],f[:,4:]

def write_features_file(filename,locs,desc):
	''' 将特征值和特征描述子保存到文件中红'''
	savetxt(filename,hstack((locs,desc)))

def plot_feature(im,locs,circle = False):
	''' 显示带有特征值的图像，输入：im(数组图像)，
	locs （每个特征的行和列，）'''

	def draw_circle(c,r):
		t = arange(0,1.01,0.01)*2*pi
		x = r*cos(t) + c[0]
		y = r*sin(t) + c[1]
		plot(x,y,'b',linewidth = 2)

	imshow(im)
	if circle:
		for p in locs:
			draw_circle(p[:2],p[2])
	else:
		plot(locs[:,0],locs[:,1],'ob')
	axis('off')

def match(desc1,desc2):
	'''根据第一幅图像的描述子，来计算第二幅图像的对应的描述子图像
	返回值：对应第二幅图像的下角标
	'''

	desc1 = array([d/linalg.norm(d) for d in desc1])

	desc2 = array([d/linalg.norm(d) for d in desc2])
	desc_ratio = 0.6

	desc1_size = desc1.shape
	matchscores = zeros((desc1_size[0],1),'int')
	#计算desc2 的转置
	desc2T = desc2.T

	for i in range(desc1_size[0]):
		dotprods = dot(desc1[i,:],desc2T)
		#进行反余弦操作，进行反排序，返回第二幅图像的特征索引

		indx = argsort(arccos(dotprods))

		# 检查紧邻的角度是否为最小的角度

		if arccos(dotprods)[indx[0]]<arccos(dotprods)[indx[1]]*desc_ratio:
			matchscores[i] = int(indx[0])
	return matchscores

def match_twoside(desc1,desc2):
	''' 双向对称版本的match()'''

	match12 = match(desc1,desc2)
	match21 = match(desc2,desc1)

	ndx_12 = match12.nonzero()[0]
	#去除不对称的匹配

	for n in ndx_12:
		if match21[int(match12[n])] != n:
			match12[n] =0

	return match12








