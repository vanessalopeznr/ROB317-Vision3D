import numpy as np
import cv2

print("Version d'OpenCV: ",cv2. __version__)

# Ouverture de l'image
PATH_IMG = './Images_Homographie/'

img = np.uint8(cv2.imread(PATH_IMG+"Pompei.jpg"))

(h,w,c) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"couleurs")

def normalisation_points(X_init):
	Transformation = np.array([[2/h, 0, -1],
									[0, 2/w, -1],
									[0,   0, 1]],np.float32)
	points_norm = []
	for x,y in X_init:
		point = np.array([x,y,1],np.float32) #Vector de 3x1
		point_trans = np.dot(Transformation, point)
		points_norm.append(point_trans)
	

def select_points(event, x, y, flags, param):
	global points_selected,X_init
	global img,clone
	if (event == cv2.EVENT_FLAG_LBUTTON):
		x_select,y_select = x,y
		points_selected += 1
		cv2.circle(img,(x_select,y_select),8,(0,255,255),1)
		cv2.line(img,(x_select-8,y_select),(x_select+8,y_select),(0,255,0),1)
		cv2.line(img,(x_select,y_select-8),(x_select,y_select+8),(0,255,0),1)
		X_init.append( [x_select,y_select] )
	elif event == cv2.EVENT_FLAG_RBUTTON:
		points_selected = 0
		img = clone.copy()
		
clone = img.copy()
points_selected = 0
X_init = []
cv2.namedWindow("Image initiale")
#Comentar la sgnte linea para poner los puntos desde codigo
#cv2.setMouseCallback("Image initiale",select_points)

X_init = [[144,  46],
[112, 254],
[369, 261],
[355, 47]]

points_selected = 4

while True:
	cv2.imshow("Image initiale",img)
	key = cv2.waitKey(1) & 0xFF
	if (key == ord("q")) & (points_selected >= 4):
		break
		
# Conversion en array numpy
X_init = np.asarray(X_init,dtype = np.float32) 		
print("X_init =",X_init)
X_final = np.zeros((points_selected,2),np.float32)
'''
for i in range(points_selected):
	string_input = "Correspondant de {} ? ".format(X_init[i])
	X_final[i] = input(string_input).split(" ",2)
'''
X_final = np.array([[50,50],
		[50,350],
		[350,350],
		[350,50]],np.float32)

print("X_final =",X_final)
X_initinit = np.array([[0,0],
		[500,0],
		[500,333],
		[333,0]],np.float32)

mean, std = np.mean(X_initinit, 0), np.std(X_initinit)

# define similarity transformation
# no rotation, scaling using sdv and setting centroid as origin
Transformation = np.array([[std/np.sqrt(2), 0, mean[0]],
							[0, std/np.sqrt(2), mean[1]],
							[0,   0, 1]],np.float32)
X_initnor=np.hstack((X_init,np.ones((X_init.shape[0],1))))
# apply transformation on data points
Transformation = np.linalg.inv(Transformation)
X_init = np.dot(Transformation, X_initnor.T)

'''
# Normalisation des points
X_init = np.hstack((X_init,np.ones((X_init.shape[0],1))))
X_final = np.hstack((X_final,np.ones((X_final.shape[0],1))))
T = np.array([[2/h, 0, 1],
		[0, 2/w, 1],
		[0, 0, 1]],np.float32)
X_init = np.dot(X_init,T)
X_final = np.dot(X_final,T)
'''
# Votre code d'estimation de H ici
# Compute M avec chaque correspondance
A = np.zeros((2*points_selected,9),np.float32)
print(A.shape)
A = []
for i in range(len(X_init)):
	ax = [-X_init[i][0], -X_init[i][1], -1, 0, 0, 0, X_final[i][0]*X_init[i][0], X_final[i][0]*X_init[i][1], X_final[i][0]]
	ay = [0, 0, 0, -X_init[i][0], -X_init[i][1], -1, X_final[i][1]*X_init[i][0], X_final[i][1]*X_init[i][1], X_final[i][1]]
	A.append(ax)
	A.append(ay)

# Convert le list en matrix
A = np.asarray(A,dtype = np.float32)
np.set_printoptions(suppress=True)

# Obtain SVD
U, S, V = np.linalg.svd(A)
print("SVD: ", V, V.shape)

# Determine H
H=V[-1,:]
H = np.reshape(H,(3,3))
print("H: ", H, H.shape)
H = H/V[-1, -1] # Normalize (H divido el ultimo valor)
print("H Normalizada: ", H, H.shape)
# Se puede interpretar la homografia: la ultima fila y columna tiene que ser cerca a 1 y se pueden ver la interpretacion de H en las diapos para decirme si hice una rotacio, translacion, etc.
denorm = np.dot(Transformation.T,H)
denorm = np.dot(denorm,Transformation)

img_warp = cv2.warpPerspective(clone, denorm, (w,h))
cv2.imshow("Image rectifiee",img_warp)
cv2.waitKey(0)
