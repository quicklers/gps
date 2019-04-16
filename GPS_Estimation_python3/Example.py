
#importar librerias necesarias
from matplotlib.pyplot import plot, legend, show, title
import matplotlib.pyplot as plt  # importa la libreria para hacer graficas


#################################### ejemplo###########################
### PREDICTION STEP


def kf_predict(X, P, A, Q, B, U):
    ABC=dot(B, U)
    X = dot(A, X) + ABC
    print(ABC)
    P = dot(A, dot(P, A.T)) + Q
    return(X, P)

### UPDATE STEP
from numpy.linalg import det


def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K,(Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return(X, P, K, IM, IS, LH)


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
#time step of mobile movement
dt = 0.1
#Initialization of state matrices
X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q = eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))
# Measurement matrices
Y = array([[X[0,0] + abs(random.random())], [X[1,0] + abs(random.random())]])
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = eye(Y.shape[0])
# Number of iterations in Kalman Filter
N_iter = 50
estim = zeros((N_iter, 2))
medidos = zeros((N_iter, 2))
trayreal = zeros((N_iter, 2))


# Applying the Kalman Filter
for i in arange(0, N_iter):
    (X, P) = kf_predict(X, P, A, Q, B, U)
    trayreal[i, 0] = X[0, 0] #real X
    trayreal[i, 1] = X[1, 0] #real Y
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
    Y = array([[X[0,0] + abs(0.1 * random.random())],[X[1, 0] + abs(0.1 * random.random())]])
    estim[i,0] = X[0,0] #estimacion X
    estim[i,1] = X[1,0] # estimacion Y
    medidos[i, 0] = Y[0, 0] #medicion X
    medidos[i, 1] = Y[1, 0] #medicion Y

    #plt.plot(Y[0,0],Y[1,0], '.' )  # grafica los datos medidos

plot(trayreal[:, 0], trayreal[:, 1], 'b-', label ='Trayecto Real')  # grafica los datos estimados en linea continua verde
plot(medidos[:, 0], medidos[:, 1], 'r.', label ='Mediciones')  # grafica los datos estimados en linea continua verde
plot(estim[:, 0], estim[:, 1], 'g-', label ='Trayecto Estimado')  # grafica los datos estimados en linea continua verde
legend() # introduce un cuadro con las leyendas
title('Trayecto') # pone titulo a la grafica
show() # muestra la grafica