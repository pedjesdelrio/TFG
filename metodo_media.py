from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


import numpy as np
import pandas as pd
import statistics as st

import math

#Definimos el número de landmaks y coordenadas con las que se va a operar
n_landmarks = 10
n_coordenadas = n_landmarks * 3


#Función para el alineamiento de dos conjuntos de puntos 3D (A,B)
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    '''if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T'''

    t = -R @ centroid_A + centroid_B

    return R, t

#Calcula el centro de masas para un conjunto de puntos dado.
def centroMasa(datos):
	count = 0
	Xs = []
	Ys = []
	Zs = []
	lista = datos.tolist()
	for i in range(n_landmarks*2):
		Xs.append(lista[count])
		Ys.append(lista[count+1])
		Zs.append(lista[count+2])
		count = count+3

	CM = [st.mean(Xs), st.mean(Ys), st.mean(Zs)]
	return CM

#Traslada a un conjunto de puntos.
def translacion(centroMasa, datos):
	count2 = 0
	coordenadas = []
	lista2 = datos.tolist()
	for i in range(n_landmarks):
		coordenadas.append(lista2[count2]-centroMasa[0])
		coordenadas.append(lista2[count2+1]-centroMasa[1])
		coordenadas.append(lista2[count2+2]-centroMasa[2])
		count2 = count2+3
	return coordenadas

#Calcula la distancia Euclídea entre dos puntos.
def euclidea(cx ,cy, cz, fx, fy, fz):
	dist = math.sqrt(pow((fx-cx),2) + pow((fy-cy),2) + pow((fz-cz),2))
	return dist

#Función que entrena un MLPRegressor con los datos (Xtrain, Ytrain) y devuelve la estimación para Xtest
def redneuronal(Xtrain, Ytrain, Xtest):
	regr = MLPRegressor(hidden_layer_sizes=(5,2), max_iter=1000000, random_state=450, solver='lbfgs', activation="identity").fit(Xtrain, Ytrain)
	resultado = regr.predict(Xtest)
	return resultado

#Función que entrena un RF con los datos (Xtrain, Ytrain) y devuelve la estimación para Xtest
def randomforest(Xtrain, Ytrain, Xtest):
	regr = RandomForestRegressor(n_estimators=250, max_features=15, min_samples_leaf=5, random_state=450).fit(Xtrain, Ytrain)
	resultado = regr.predict(Xtest)
	return resultado

#Función que entrena un PLS con los datos (Xtrain, Ytrain) y devuelve la estimación para Xtest
def PLS(Xtrain, Ytrain, Xtest):
	regr = PLSRegression(n_components=17).fit(Xtrain, Ytrain)
	resultado = regr.predict(Xtest)
	return resultado


#---------------------OBTENEMOS LOS DATOS DEL FICHERO Y CENTRAMOS LOS MODELOS 3D------------------------------

#Leemos el fichero completo.
alldata = pd.read_excel ("RAW_3D_Full_data_PG2011.xlsx", engine='openpyxl')

#Nos quedamos con los landmarks, tanto craneométricos como cefalométricos, que nos interesan
landmarks = alldata[['xCg2', 'yCg2', 'zCg2',
			 'xNn2', 'yNn2', 'zNn2', 
             'xBgn2', 'yBgn2', 'zBgn2', 
             'xBpg2', 'yBpg2', 'zBpg2',
             'xNalG', 'yNalG', 'zNalG',
             'xNalD', 'yNalD', 'zNalD',
             'xYenG', 'yYenG', 'zYenG',
             'xYenD', 'yYenD', 'zYenD',
             'xYexG', 'yYexG', 'zYexG',
             'xYexD', 'yYexD', 'zYexD',
             'xCg', 'yCg', 'zCg', 
             'xNn', 'yNn', 'zNn', 
             'xBgn', 'yBgn', 'zBgn', 
             'xBpg', 'yBpg', 'zBpg',
             'xNapG', 'yNapG', 'zNapG',
             'xNapD', 'yNapD', 'zNapD',
             'xYdG', 'yYdG', 'zYdG',
             'xYdD', 'yYdD', 'zYdD',
             'xYekG', 'yYekG', 'zYekG',
             'xYekD', 'yYekD', 'zYekD']]


#Eliminamos aquellas filas en las que falta algún dato.
NoNAN = landmarks.dropna()
NoNAN = NoNAN[:-1]

#Obtenemos el número de muestras con el que disponemos

n_muestras = NoNAN.shape[0]

#Interamos por todas las filas para calcular el centro de masas de cada muestra.
CMs = []
for i in range (n_muestras):
	CMs.append(centroMasa(NoNAN.iloc[i,0:n_coordenadas*2]))

#Dividimos en landmarks craneales y faciales.
landmarksC = []
landmarksF = []
for i in range (n_muestras):
	landmarksC.append(translacion(CMs[i], NoNAN.iloc[i,n_coordenadas:n_coordenadas*2]))
	landmarksF.append(translacion(CMs[i], NoNAN.iloc[i,0:n_coordenadas]))


#--------------------------------------------ALINEAMOS AHORA LOS CRÁNEOS---------------------------------


#Es necesario reestructurar como están almacenadas las coordenadas de cada individuo
#para poder aplicar el algoritmo de alineamiento
landmarks = []
for i in range(n_muestras):
	landmarks.append(landmarksC[i] + landmarksF[i])

nuevos_landmarks = []
contador = 0
for individuo in landmarks:
	x,y,z = [],[],[]
	for coordenada in individuo:
		if contador == 0:
			x.append(coordenada)
		elif contador == 1:
			y.append(coordenada)
		elif contador == 2:
			z.append(coordenada)
		contador+=1
		if contador == 3:
			contador = 0
	nuevos_landmarks.append([x,y,z])



nuevos_landmarks = np.array(nuevos_landmarks)

tamanio = []
for i in range(len(nuevos_landmarks.shape)):
	tamanio.append(nuevos_landmarks.shape[i])

rotados = np.empty(tamanio)


#Se escoge el primer individuo como destino, todos los demás se alinearan a él
destino = nuevos_landmarks[0]
for i in range(len(nuevos_landmarks)):
	origen = nuevos_landmarks[i]
	#Se calcula la matriz de rotación y se aplica
	ret_R, ret_t = rigid_transform_3D(origen, destino)
	rotado = (ret_R@origen)
	rotados[i] = rotado


rotados = rotados.tolist()

#Finalmente volvermos a estructurar los datos como se tenían al comienzo
a_entrenar = []
for elemento in rotados:
	individuo = []
	for i in range(len(elemento[0])):
		individuo.append(elemento[0][i])
		individuo.append(elemento[1][i])
		individuo.append(elemento[2][i])
	a_entrenar.append(individuo)


landmarksC = []
landmarksF = []

for elemento in a_entrenar:
	landmarksC.append(elemento[0:n_coordenadas])
	landmarksF.append(elemento[n_coordenadas:n_coordenadas*2])

#------------------------------------------CALCULAMOS EL GROSOR DEL TEJIDO BLANDO---------------------------

FSTDs = []
FSTD = []
#Calculamos ahora el FSTD
for i in range(len(landmarksC)):
	contador2 = 0
	for j in range(n_landmarks):
		FSTD.append(euclidea(landmarksC[i][contador2], landmarksC[i][contador2+1], landmarksC[i][contador2+2], landmarksF[i][contador2], landmarksF[i][contador2+1], landmarksF[i][contador2+2]))
		contador2 = contador2+3
	FSTDs.append(FSTD)
	FSTD = []


medias=[]
#Calculamos la media para cada punto.
for i in range(n_landmarks):
	suma=0
	for j in range(len(FSTDs)):
		suma += FSTDs[j][i]
	suma/=len(FSTDs)
	medias.append(suma)



reales=[[],[],[],[],[],[],[],[],[],[]]
estimado=[[],[],[],[],[],[],[],[],[],[]]


#La estimación para un landmark será el valor promedio calculado previamente
for i in range(n_landmarks):
	reales[i] += [FSTDs[j][i] for j in range(len(FSTDs))]
	for j in range(len(FSTDs)):
		estimado[i].append(medias[i])

rmse = []
mae = []

for i in range(len(reales)):
	rmse.append(math.sqrt(mean_squared_error(reales[i], estimado[i])))
	mae.append(mean_absolute_error(reales[i], estimado[i]))



#Calculamos error relativo
error_relativo = []
for i in range(len(reales)):
	aux = 0
	for j in range(len(reales[i])):
		aux += abs((estimado[i][j] - reales[i][j])/reales[i][j])*100
	aux /= len(reales[i])
	error_relativo.append(aux)



table_vals = [rmse[:4],mae[:4],error_relativo[:4]]

fig, ax = pyplot.subplots(1,1)
row_labels= ["RMSE","MAE","Error Relativo"]
col_labels = ['Cg','Nn','Bgn','Bpg']
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_vals,loc="center",rowLabels=row_labels,colLabels=col_labels)

pyplot.show()


table_vals = [rmse[4:7],mae[4:7],error_relativo[4:7]]

fig, ax = pyplot.subplots(1,1)
row_labels= ["RMSE","MAE","Error Relativo"]
col_labels = ['NapG','NapD','YdG']
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_vals,loc="center",rowLabels=row_labels,colLabels=col_labels)

pyplot.show()

table_vals = [rmse[7:10],mae[7:10],error_relativo[7:10]]

fig, ax = pyplot.subplots(1,1)
row_labels= ["RMSE","MAE","Error Relativo"]
col_labels = ['YdD','YekG','YekD']
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_vals,loc="center",rowLabels=row_labels,colLabels=col_labels)

pyplot.show()