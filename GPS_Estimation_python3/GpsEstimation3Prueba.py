# importar librerias necesarias
import pandas as pd  # importa la libreria pandas y la nombra como pd

datosReales = pd.read_csv('RutaCortaReal.csv',
                          header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0
datosGPS = pd.read_csv('RutaCorta.csv',
                       header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0

# importar librerias necesarias
import matplotlib.pyplot as plt  # importa la libreria para hacer graficas


#################################### ejemplo###########################
### PREDICTION STEP


def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return (X, P)


### UPDATE STEP
from numpy.linalg import det


def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y - IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X, P, K, IM, IS, LH)


def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1]) - M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X - M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0], E[0])


####### EJEMPLO PRACTICO
from numpy import *
from numpy.linalg import inv

t = 1.0
# time step of mobile movement
dt = 0.1
# Initialization of state matrices
X = array([[datosReales['lat'].ix[0]], [0.0], [datosReales['lon'].ix[0]], [0.0]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, t, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]])
Q = eye(X.shape[0])
B = array([[(t ** 2) / 2.0], [t], [(t ** 2) / 2.0], [t]])
U = 0.0
# Measurement matrices
Y = array([[datosReales['lat'].ix[0]], [datosGPS['speed'].ix[0]], [datosReales['lon'].ix[0]], [datosGPS['speed'].ix[0]]])
H = array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
R = eye(Y.shape[0])
# Number of iterations in Kalman Filter
N_iter = datosGPS.__len__() / dt
estim = zeros((N_iter.__int__(), 4))
medidos = zeros((datosGPS.__len__(), 4))
trayreal = zeros((N_iter.__int__(), 4))
aux = 0

# Applying the Kalman Filter
for i in arange(0, N_iter.__int__()):

    if i == ((N_iter / datosGPS.__len__()) * aux):
        Y = array([[datosReales['lat'].ix[aux]+(random.random()/10000)], [0.0], [datosReales['lon'].ix[aux]+(random.random()/100000)], [0.0]])
        #Y = array([[datosGPS['lat'].ix[aux]], [0.0], [datosGPS['lon'].ix[aux]], [0.0]])
        #X = array([[datosReales['lat'].ix[aux]], [0.0], [datosReales['lon'].ix[aux]], [0.0]])

        medidos[aux, 0] = Y[2, 0]  # medicion posicion X
        medidos[aux, 1] = Y[0, 0]  # medicion posicion Y
        medidos[aux, 2] = Y[1, 0]  # medicion velocidad X
        medidos[aux, 3] = Y[3, 0]  # medicion velocidad Y
        aux = aux + 1
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)

    estim[i, 0] = X[2, 0]  # estimacion X
    estim[i, 1] = X[0, 0]  # estimacion Y
    estim[i, 2] = X[1, 0]  # medicion velocidad X
    estim[i, 3] = X[3, 0]  # medicion velocidad Y

plt.plot(datosReales['lon'], datosReales['lat'], 'b-',
         label='Trayecto Real')  # grafica los datos reales en linea continua verde
plt.plot(estim[:, 0], estim[:, 1], 'g*',
         label='Trayecto Estimado')  # grafica los datos estimados en linea continua verde
plt.plot(medidos[:, 0], medidos[:, 1], 'r.', label='Mediciones')  # grafica los datos estimados en linea continua verde
plt.legend()  # introduce un cuadro con las leyendas
plt.xlabel('Longitud')  # nombra el eje x
plt.ylabel('Latitud')  # nombra el eje y
plt.title('Position')  # pone titulo a la grafica
plt.show()  # muestra la grafica
plt.clf()
plt.plot(datosGPS['speed'], datosGPS['speed'], 'b-',
         label='Trayecto Real')  # grafica los datos reales en linea continua verde
plt.plot(estim[:, 2], estim[:, 3], 'g*',
         label='Trayecto Estimado')  # grafica los datos estimados en linea continua verde
plt.plot(medidos[:, 2], medidos[:, 3], 'r.', label='Mediciones')  # grafica los datos estimados en linea continua verde
plt.legend()  # introduce un cuadro con las leyendas
plt.xlabel('Longitud')  # nombra el eje x
plt.ylabel('Latitud')  # nombra el eje y
plt.title('Velocidad')  # pone titulo a la grafica
plt.show()  # muestra la grafica