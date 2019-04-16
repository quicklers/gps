
#importar librerias necesarias
import pandas as pd   #importa la libreria pandas y la nombra como pd
import matplotlib.pyplot as plt #importa la libreria para hacer graficas
import math #importa libreria para realizar operaciones matematicas


#declaracion de variables
longitud_before = 0 #longitud del estado anterior
longitud_now = 0 #longitud del estado actual
latitud_before = 0 #latitud del estado anterior
latitud_now = 0 #latitud del estado actual
punto = 0 #punto nuevo que indica el GPS
index = 0 #variable para conocer si es dato de GPS o de IMU


#variables de posiciones GPS
posX_GPS = 0
posY_GPS = 0
posZ_GPS = 0

#variables para calcular la distancia entre dos puntos dela tierra (formula de haversine)
distancia_total = 0 #distancia total recorrida
rad = math.pi/180 #multiplo para transformar de rados a radianes
R = 6372.795477598 #radio de la tierra, en kilometros
#variables para el angulo de seguimiento del GPS
track_angle = 0
import seaborn as sns
datos = pd.read_csv('RutaCorta.csv',header = 0) #importa el archivo .csv e indica que el encabezado esta en la fila 0
punto = 1

##calculo de la distancia de haversine
for punto in range(datos.__len__()-1):

    longitud_before = (datos['lon'].ix[0])  # columna de longitud y fila 0
    longitud_now = (datos['lon'].ix[punto])  # columna de longitud y fila 1
    latitud_before = (datos['lat'].ix[0])  # columna de latitud y fila 0
    latitud_now = (datos['lat'].ix[punto])  # columna de latitud y fila 1

    dlat = latitud_now - latitud_before  # delta de la latitud (diferencia de distancia entre los dos puntos de la latitud)
    dlon = longitud_now - longitud_before  # delta de la longitud (diferencia de distancia entre los dos puntos de la longitud)

        # aplicando la formula de haversine
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * latitud_before) * math.cos(rad * latitud_now) * (
        math.sin(rad * dlon / 2)) ** 2
    distancia = 2 * R * math.asin(
        math.sqrt(a))  # calcula la distancia entre los dos puntos(inicial y final) en kilometros

        # distancia_total = distancia_total + distancia #calcula la distancia total recorrida en kilometros
        # print("distancia = " + distancia.__str__())

        # calculo del angulo de seguimiento

    track_angle = math.atan(
        math.sin(rad * dlon) * math.cos(rad * latitud_now) * math.cos(rad * latitud_before) * math.sin(
            rad * latitud_now) - math.sin(rad * latitud_before) * math.cos(rad * latitud_now) * math.cos(
            rad * dlon))
    track_angle = track_angle * (180 / (math.pi))
        # print("angulo = "+track_angle.__str__())

        # calculo de posiciones GPS
    posX_GPS = distancia * math.sin(track_angle)
    posY_GPS = distancia * math.cos(track_angle)
    posZ_GPS = datos['elevation'].ix[punto]

    print (posX_GPS)
    print (posY_GPS)
    print (posZ_GPS)

    plt.plot(longitud_now, latitud_now,'.')  # grafica los puntos de longitu y latitud y cada punto lo indica como '.'

    punto = punto + 1
    #fin del ciclo for y del calcul de la distancia con la formula de haversine


plt.xlabel('Longitud') # nombra el eje x
plt.ylabel('Latitud') # nombra el eje y
plt.show() # muestra la grafica
plt.plot(datos['lon'], datos['lat'],'-')
plt.show() # muestra la grafica




