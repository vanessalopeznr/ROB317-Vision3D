import numpy as np
import cv2

print("Version d'OpenCV: ",cv2. __version__)

# Ouverture de l'image
PATH_IMG = './Images_Homographie/'

img = np.uint8(cv2.imread(PATH_IMG+"Pompei.jpg"))

(h,w,c) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"couleurs")

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

X_init = [[122, 26],
 [ 75, 283],
 [406, 298],
 [382, 30]]
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

X_final = np.array([[0,0],
		[0,250],
		[250,250],
		[250,0]],np.float32)

Transformation = np.array([[2/h, 0, -1],
							[0, 2/w, -1],
							[0,   0, 1]],np.float32)

points_norm_i = []
points_norm_f = []

#Normalisation des points
for i,f in zip(X_init,X_final):
		point_i = np.array([i[0],i[1],1],np.float32) #Vector de 3x1
		point_f = np.array([f[0],f[1],1],np.float32) #Vector de 3x1
		point_trans_i = np.dot(Transformation, point_i)
		point_trans_f = np.dot(Transformation, point_f)
		points_norm_i.append(point_trans_i)
		points_norm_f.append(point_trans_f)

#print("Points : ", points_norm, points_norm_x)

X_init = points_norm_i
X_final = points_norm_f

# Votre code d'estimation de H ici
# Compute M avec chaque correspondance
A = np.zeros((2*points_selected,9),np.float32)
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

# Determine H
H=V[-1,:]
H = np.reshape(H,(3,3))
H = H/V[-1, -1] # Normalize (H divido el ultimo valor)

# Denormalisation de H
denorm = np.dot(np.linalg.inv(Transformation),H)
denorm = np.dot(denorm,Transformation)

img_warp = cv2.warpPerspective(clone, denorm, (w,h))
cv2.imshow("Image rectifiee",img_warp)
cv2.waitKey(0)
