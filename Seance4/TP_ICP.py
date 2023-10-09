# TP MAREVA Nuages de Points et Mod�lisation 3D - Python - FG 24/09/2020
# coding=utf8

# Import Numpy
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

# Import functions from scikit-learn : KDTree
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
# utils.ply est le chemin relatif utils/ply.py


def read_data_ply(path):
# Lecture de nuage de points sous format ply
    '''
    Lecture de nuage de points sous format ply
    Inputs :
        path = chemin d'acc�s au fichier
    Output :
        data = matrice (3 x n)
    '''
    data_ply = read_ply(path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))
    return(data)

def write_data_ply(data,path):
    '''
    Ecriture de nuage de points sous format ply
    Inputs :
        data = matrice (3 x n)
        path = chemin d'acc�s au fichier
    '''
    write_ply(path, data.T, ['x', 'y', 'z'])
    
def show3D(data,nom):
    '''
    Visualisation de nuages de points avec MatplotLib'
    Input :
        data = matrice (3 x n)
    '''
    #plt.cla()
    # Aide en ligne : help(plt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], '.')
    #ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
    #plt.axis('equal')
    plt.title(nom) 
    plt.show()

def show3D2(data,data_aligned,nom):
    '''
    Visualisation de nuages de points avec MatplotLib'
    Input :
        data = matrice (3 x n)
    '''
    #plt.cla()
    # Aide en ligne : help(plt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], '.',color='blue')
    ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.',color='red')
    #plt.axis('equal')
    plt.title(nom)
    plt.legend(["Original","Changed"])  
    plt.show()

# Fonction pour decimer un nuage de points
def decimate(data,k_ech):
    met=True
    if met==True:    
        # 1ere methode : boucle for
        n=data.shape[1]
        n_ech=int(n/k_ech)

        dec = np.vstack(data[:,0])
        for i in range(1,n_ech):
            # Xi = vecteur du rang k_ech*i (utiliser np.vstack)
            Xi=np.vstack(data[:,k_ech*i])
            # concatener Xi a decimated en utilisant np.hstack
            dec = np.hstack((dec, Xi))           
            
    else:
        #decimated = sous-tableau des colonnes espac�es de k_ech
        res=[]
        i=0
        print(data.shape)
        while i < data.shape[1]:
            res.append(data[:,i])
            i += k_ech
        dec = np.stack(res, axis=1)
        
    return(dec)


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
         ref = (d x N) matrix where "N" is the number of point and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # Barycenters
    # définir les baycentres ref_center et data_center
    data_center=np.mean(data, axis=1).reshape(3,1)
    ref_center=np.mean(ref, axis=1).reshape(3,1)

    # Centered clouds
    # calculer les nuages de points centrés ref_c et data_c
    data_c = data - data_center
    ref_c = ref - ref_center

    # H matrix
    # calculer la matrice H
    H=np.dot(data_c,ref_c.T)

    # SVD on H
    # calculer U, S, et Vt en utilisant np.linalg.svd
    # Decomposer H en valeurs singulières
    U, S, Vt = np.linalg.svd(H)

    # Checking R determinant
    # si le déterminant de U est -1, prendre son opposé
    np.linalg.det(U)
    # Getting R and T
    R=np.dot(Vt.T,U.T)
    # calculer R et T
    T=ref_center - np.dot(R,data_center)

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    search_tree = KDTree(ref.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    for i in range(max_iter):

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned.T, return_distance=True)

        # Compute average distance 
        # calculer la distance moyenne entre les points
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        # indices.ravel() = indices sous forme de vecteur
        R, T = best_rigid_transform(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T


    return data_aligned, R_list, T_list, neighbors_list, RMS_list




#
#           Main
#       \**********/
#

if __name__ == '__main__':


    # Fichiers de nuages de points
    bunny_o_path = 'data/bunny_original.ply'
    bunny_p_path = 'data/bunny_perturbed.ply'
    bunny_r_path = 'data/bunny_returned.ply'
    NDC_o_path = 'data/Notre_Dame_Des_Champs_1.ply'
    NDC_p_path = 'data/Notre_Dame_Des_Champs_2.ply'
    NDC_r_path = 'data/Notre_Dame_Des_Champs_returned.ply'

    # Lecture des fichiers
    bunny_o=read_data_ply(bunny_o_path)                    
    bunny_p=read_data_ply(bunny_p_path)
    NDC_o=read_data_ply(NDC_o_path)
    NDC_p=read_data_ply(NDC_p_path)

    # Visualisation du fichier d'origine
    if False:
        show3D(bunny_o,"Original")

    # Transformations : d�cimation, rotation, translation, �chelle
    # ------------------------------------------------------------
    if False:
        # D�cimation        
        decimated = decimate(bunny_o,10)
        
        # Visualisation sous Python et par �criture de fichier
        show3D2(bunny_o,decimated,"Decimated")
        
        # Visualisation sous CloudCompare apr�s �criture de fichier
        write_data_ply(decimated,bunny_r_path)
        # Puis ouvrir le fichier sous CloudCompare pour le visualiser

    if False:
        decimated = decimate(NDC_o,1000)
        show3D2(NDC_o,decimated,"Decimated")
        write_data_ply(decimated,NDC_r_path)

    if False:        
        # Translation
        # translation = définir vecteur [0, -0.1, 0.1] avec np.array et reshape
        translation=np.array([0, -0.1, 0.1]).reshape(3,1)
        points=bunny_o + translation
        show3D2(bunny_o,points,"Translated")
        
        # Find the centroid of the cloud and center it
        #centroid = barycentre - utiliser np.mean(points, axis=1) et reshape
        centroid=np.mean(points, axis=1).reshape(3,1)
        points_cent = points - centroid
        show3D2(bunny_o,points_cent,"Centered")
        
        # Echelle
        # points = points divisés par 2
        points= points/2
        show3D2(bunny_o,points,"Scaled")
        
        # Define the rotation matrix (rotation of angle around z-axis)
        # angle de pi/3,
        # définir R avec np.array et les cos et sin.
        R=np.array([[np.cos(np.pi/3), -np.sin(np.pi/3), 0],
                    [np.sin(np.pi/3), np.cos(np.pi/3), 0],
                    [0, 0, 1]])
        
        # Apply the rotation
        points=np.dot(R,bunny_o)
        # centrer le nuage de points        
        # appliquer la rotation - utiliser la fonction .dot
        # appliquer la translation opposée
        show3D2(bunny_o,points,"Rotated")
        

    # Meilleure transformation rigide (R,Tr) entre nuages de points
    # -------------------------------------------------------------
    if True:
        # CONEJO
        show3D2(bunny_o,bunny_p,"Dataset")
        
        # Find the best transformation
        R, Tr = best_rigid_transform(bunny_p, bunny_o)
        
        
        # Apply the tranformation
        opt = R.dot(bunny_p) + Tr
        bunny_r_opt = opt
        
        # Show and save cloud
        show3D2(bunny_o,bunny_r_opt,"Changed")
        write_data_ply(bunny_r_opt,bunny_r_path)
        
        
        # Get average distances
        distances2_before = np.sum(np.power(bunny_p - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))
        
        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
        '''

        show3D2(NDC_o,NDC_p,"Dataset")
        
        # Find the best transformation
        R, Tr = best_rigid_transform(NDC_p, NDC_o)
        
        
        # Apply the tranformation
        opt = R.dot(NDC_p) + Tr
        NDC_r_opt = opt
        
        # Show and save cloud
        show3D2(NDC_o,NDC_r_opt,"Changed")
        write_data_ply(NDC_r_opt,NDC_r_path)
        
        
        # Get average distances
        distances2_before = np.sum(np.power(NDC_p - NDC_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(NDC_r_opt - NDC_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))
        
        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
        '''
    # Test ICP and visualize
    # **********************
    if False:

        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        plt.plot(RMS_list)
        plt.show()





