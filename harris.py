from numpy import *
from PIL import Image
from pylab import *
from scipy.ndimage import filters

''' 在一副灰度图像当中，对每一个像素计算harris角点
及其响应函数'''

# 计算倒数

def compute_harris_response(im,sigma = 3):
	imx = zeros(im.shape)
	filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
	imy = zeros(im.shape)
	filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)

	#计算harris 矩阵的分量
	Wxx = filters.gaussian_filter(imx*imx,sigma)
	Wxy = filters.gaussian_filter(imx*imy,sigma)
	Wyy = filters.gaussian_filter(imy*imy,sigma)

	#计算特征值和迹

	Wdet = Wxx*Wyy - Wxy**2
	Wtr = Wxx + Wyy

	return Wdet/Wtr

def get_harris_points(harrisim,min_dist = 2,threshold = 0.01):
	'''从一副Harris响应图像中返回角点min_dist 为分割角点
	和图像边界的最小像素点数'''

	corner_threshold = harrisim.max() * threshold
	harrisim_t = (harrisim > corner_threshold)*1

	#得到候选的坐标

	coords = array(harrisim_t.nonzero()).T
	#以及他们的HARRIS响应值
	candidate_values = [harrisim[c[0],c[1]] for c in coords]

	#对候选点的响应值进行排序
	index = argsort(candidate_values)
	#将可行点的位置保存到数组中
	allowed_locations = zeros(harrisim.shape)
	allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
	#按照最小距离的原则，选择最佳的HARRIS点
	filtered_coords = []
	for i in index:
		if allowed_locations[coords[i,0],coords[i,1]] == 1:
			filtered_coords.append(coords[i])
			allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
	print(filtered_coords)
	return filtered_coords

def plot_harris_points(image,filtered_coords):
	figure()
	gray()
	imshow(image)
	plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
	show()

def get_descripter(image,filtered_coords,wid=5):
	''' 对于每个返回的点，返回周边2*wid+1的像素'''
	desc = []
	for coords in filtered_coords:
		patch = image[coords[0]-wid:coords[0]+wid+1].flatten()
		desc.append(patch)
	return desc

def match(desc1,desc2,threshold = 0.5):
	'''对于第一幅图像中的每一个角点描述子，使用归一化互相关
	选取它在第二幅图中的匹配角点'''

	n = len(desc1)
	#计算点对的距离
	d = -ones((len(desc1),len(desc2)))
	for i in range(len(desc1)):
		for j in range(len(desc2)):
			d1 = (desc1[i]-mean(desc1[i]))/std(desc1[i])
			d2 = (desc2[j] - mean(desc2[j]))/std(desc2[j])
			ncc_value = sum(d1*d2)/(n-1)
			if(ncc_value>threshold):
				d[i,j] = ncc_value
	ndx = argsort(-d)
	matchscores = ndx[:,0]
	return matchscores

def match_twoside(desc1,desc2,threshold = 0.5):
	''' 这个是两边对称版本的match()'''
	matches_12 = match(desc1,desc2,threshold)
	matches_21 = match(desc2,desc1,threshold)

	ndx_12 = where(matches_12 >= 0)[0]
	#去掉非对称的匹配

	for n in ndx_12:
		if matches_21[matches_12[n]] != n:
			matches_21[n] = -1

	return matches_12

def appendimages(im1,im2):
	''' 返回将两幅图像并排拼接成的一副新图像'''
	#选取具有最小行数的图像，然后填充足够的空行

	row1 = im1.shape[0]
	row2 = im2.shape[0]

	if row1<row2:
		im1 = concatenate((im1,zeros((row2 - row1,im1.shape[1]))),axis = 0)
	elif row1 > row2:
		im2 = concatenate((im2,zeros((row1-row2,im2.shape[1]))),axis= 0)

	return concatenate((im1,im2),axis = 1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below = True):

	''' 显示一副带有连接匹配的之间连线的图片'''

	''' 输入 数组图片 locs1,2等特征位置，matchscores
	'''

	im3 = appendimages(im1,im2)
	if show_below:
		im3 = vstack((im3,im3))

		imshow(im3)

		cols = im1.shape[1]
		for i,m in enumerate(matchscores):
			if m>0:
				plot([locs1[i,0],locs2[m,0]+cols],[locs1[i,1],locs2[m,1]],'c')
				axis('off')


#下面是主程序

'''im1 = array(Image.open('./timg1.jpg').convert('L'))

im2 = array(Image.open('./timg2.jpg').convert('L'))
wid = 5
harrisim = compute_harris_response(im1,5)
filtered_coords1 = get_harris_points(harrisim,wid+1)
d1 = get_descripter(im1,filtered_coords1,wid)

harrisim = compute_harris_response(im2,5)
filtered_coords2 = get_harris_points(harrisim,wid+1)

d2 = get_descripter(im2,filtered_coords2,wid)

print('starting matching')

matches = match(d1,d2)

figure()

gray()

plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)

show()'''


'''im1 = array(Image.open('./timg1.jpg').convert('L'))

im2 = array(Image.open('./timg2.jpg').convert('L'))

harrisim = compute_harris_response(im1)
filtered_coords1 = get_harris_points(harrisim)

plot_harris_points(im1,filtered_coords1)
'''

