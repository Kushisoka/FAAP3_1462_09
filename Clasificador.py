from abc import ABCMeta,abstractmethod
import numpy as np
from math import *


class Clasificador:
  
	# Clase abstracta
	__metaclass__ = ABCMeta

	# Metodos abstractos que se implementan en casa clasificador concreto
	@abstractmethod
	# TODO: esta funcion deben ser implementadas en cada clasificador concreto
	# datosTrain: matriz numpy con los datos de entrenamiento
	# atributosDiscretos: array bool con la indicatriz de los atributos nominales
	# diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
	# de variables discretas
	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
		pass


	@abstractmethod
	# TODO: esta funcion deben ser implementadas en cada clasificador concreto
	# devuelve un numpy array con las predicciones
	def clasifica(self,datosTest,atributosDiscretos,diccionario):
		pass


	# Obtiene el numero de aciertos y errores para calcular la tasa de fallo
	# TODO: implementar
	def error(self,datos,pred):
	# Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
		return sum(map(lambda x, y: 0 if x == y else 1, datos[:,-1], pred))/(len(datos[:,-1]) + 0.0)


	# Realiza una clasificacion utilizando una estrategia de particionado determinada
	# TODO: implementar esta funcion
	def validacion(self, particionado, dataset, clasificador, seed=None):

		# Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
		# - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
		# y obtenemos el error en la particion de test i
		# - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
		# y obtenemos el error en la particion test
		errores = np.array(())
		if len(particionado.particiones) == 1:
			clasificador.entrenamiento(dataset.extraeDatos(particionado.particiones[0].indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
			dataTrain=dataset.extraeDatos(particionado.particiones[0].indicesTest)
			clases = clasificador.clasifica(dataset.extraeDatos(particionado.particiones[0].indicesTest),dataset.nominalAtributos,dataset.diccionarios)
			print(len(clases))
			return self.error(dataTrain, clases), 0
		else:
			for part in particionado.particiones:
				clasificador.entrenamiento(dataset.extraeDatos(particionado.particiones[0].indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
				dataTrain = dataset.extraeDatos(part.indicesTest)
				clases = clasificador.clasifica(dataset.extraeDatos(part.indicesTest),dataset.nominalAtributos,dataset.diccionarios)
				errores=np.append(errores,[self.error(dataTrain, clases)])
			return errores.mean(), errores.std()

	
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        self.tV = []
        self.tC = {}
        self.laplace = laplace

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        self.tC = {}
        self.tV = []
        num_Clases = len(diccionario[-1])
        i = 0
        num_rows = datostrain.shape[0]
        if num_rows == 0:
            num_rows = 0.0001
        for k in diccionario[-1].keys():
            value = diccionario[-1][k]
            self.tC[k] = datostrain[np.ix_(datostrain[:,-1] == value, (0,))].shape[0]/(num_rows +0.0)

        while i < len(diccionario)-1 :

            if atributosDiscretos[i]:
                a = np.zeros((len(diccionario[i]), num_Clases))
                for row in datostrain:
                    a[int(row[i]), int(row[-1])] += 1
                if self.laplace and np.any(a==0):
                    a+=1

            else:
                a = np.zeros((2, num_Clases))
                for k in diccionario[-1].keys():
                    a[0, int(diccionario[-1][k])] = np.mean(datostrain[np.ix_(datostrain[:, -1] == diccionario[-1][k], (i, ))])
                    a[1, int(diccionario[-1][k])] = np.var(datostrain[np.ix_(datostrain[:, -1] == diccionario[-1][k], (i ,))])

            self.tV.append(a)
            i += 1


    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):

        classes = []
        for row in datostest:
            ppost = {}
            for k in diccionario[-1].keys():
                v = diccionario[-1][k]
                a = 1
                i = 0
                while i < (len(row) - 1):
                    if atributosDiscretos[i]:
                        a *= (self.tV[i][int(row[i]), v] / sum(self.tV[i][:, v]))
                    else:
                        exp = math.exp(-(((row[i]-self.tV[i][0,v])**2)/(2.0*self.tV[i][1,v])))
                        sqrt = math.sqrt(2*math.pi*self.tV[i][1,v])
                        
                        a *= (exp/sqrt)

                    i += 1
                a = a*self.tC[k]
                ppost[k] = a

            classes.append(diccionario[-1][max(ppost,key=ppost.get)])

        return np.array(classes)

    
    
class ClasificadorVecinosProximos(Clasificador):

	def __init__(self, k = 3 ,norm = False):
		self.norm = norm
		self.mean = []
		self.std = []
		self.k = k

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		
		if self.norm:
			self.calcularMediasDesv(datostrain)
			self.datos_normalizados = self.normalizarDatos(datostrain)
		else:
			self.datostrain=datostrain
		return

	def clasifica(self, datostest, atributosDiscretos, diccionario):

		if self.norm:
			test_data = self.normalizarDatos(datostest)
			train_data = self.datos_normalizados
		else:
			test_data = datostest
			train_data = self.datostrain
			
		resultado = []

		for rows_test in test_data:
			list_error = []
			distancias = []
			for rows_train in train_data:
				suma_mse = 0

				for idx in range(len(rows_train) - 1):
					suma_mse += (rows_train[idx] - rows_test[idx])**2
					
				suma_mse = sqrt(suma_mse)
				list_error += [suma_mse]
				distancias += [(suma_mse, rows_train[-1])]

			clases = np.array([])

			for _ in range(self.k):
				minimal = min(list_error)
				for i, error in enumerate(list_error):
					if error == minimal:
						pos=i
					
				error, clase = distancias[pos]

				distancias.pop(pos)
				list_error.pop(pos)
				clases = np.append(clases, clase)

			freqs = np.unique(clases, return_counts=True)

			for i, counts in enumerate(freqs[1]):
				if counts == np.max(freqs[1]):
					pos = i

			resultado += [freqs[0][pos]]
		return np.array(resultado)

	def calcularMediasDesv(self, datostrain):

		datos_atributos = np.transpose(datostrain)[:-1]
		for datos_atributo in datos_atributos:
			self.mean += [np.mean(datos_atributo)]
			self.std += [np.std(datos_atributo)]
		return

	def normalizarDatos(self, datos):
		
		datos_atributos = np.transpose(datos)[:-1]
		datos_clase = np.transpose(datos)[-1]

		attr_norm = []
		idxatributo = 0

		for datos_atributo in datos_atributos:
			for dato in datos_atributo:
				attr_norm += [(dato - self.mean[idxatributo])/self.std[idxatributo]]
			idxatributo += 1

		attr_norm = np.array(attr_norm)
		attr_norm = np.reshape(attr_norm, (len(datos_atributos),len(datos)))

		datos_atributos = np.transpose(attr_norm)
		datos_clase = np.reshape(datos_clase,(len(datos), 1))
		datos_normalizados = np.concatenate((datos_atributos, datos_clase), axis=1)

		return datos_normalizados
    
    
    
    
class ClasificadorRegresionLogistica(Clasificador):

    def __init__(self, cteApr=1,numEp=50):
        self.numEp = numEp
        self.cteApr = cteApr
        self.w=[]

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        
        if len(self.w)!= len(diccionario):
           self. w = np.random.uniform(low=-0.5,high=0.5, size=(1,len(diccionario)))
        for i in range(self.numEp):
            for row in datostrain:
                aux = np.append([1],row[:-1])
                self.w = self.w - (self.cteApr*(self.sigmoidal(np.dot(self.w,aux))-row[-1]))*aux
            i=i+1

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        classes = []
        for row in datostest:
           aux = np.append([1], row[:-1])
           classes.append(1 if self.sigmoidal(np.dot(self.w,aux)) >= 0.5 else 0 )
        return np.array(classes)



    def sigmoidal(self,p):
        try:
           aux=1.0/(1+exp(-p))
        except OverflowError:
            aux= 0.0 
        return aux
