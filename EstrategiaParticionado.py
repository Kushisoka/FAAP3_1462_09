# coding=utf-8
from abc import ABCMeta,abstractmethod
import numpy as np
import random


class Particion:
  
  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]
    

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

	# Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
	# Devuelve una lista de particiones (clase Particion)
	def creaParticiones(self,datos, porc_train,seed=None):  

		random.seed(seed)
		#Definimos los atributos de la clase EstrategiaParticionado
		self.nombreEstrategia="ValidacionSimple"
		self.numeroParticiones=1
		self.numeroParticiones=0
		self.particiones=[]

		#Creamos una particion
		part = Particion()

		#Creamos una lista de índices de 0 a n_lineas, y los desordenamos aleatoriamente
		lista=list(range(datos.n_lineas))
		random.shuffle(lista)

		#Cogemos tantos índices para train como nos marque porc_train. El resto para test
		n_train=int(datos.n_lineas*porc_train)
		part.indicesTrain=lista[:n_train]
		part.indicesTest=lista[n_train:]
		self.particiones.append(part)

		return self.particiones

    
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
	# Crea particiones segun el metodo de validacion cruzada.
	# El conjunto de entrenamiento se crea con las nfolds-1 particiones
	# y el de test con la particion restante
	# Esta funcion devuelve una lista de particiones (clase Particion)
	def creaParticiones(self,datos,k,seed=None):

		random.seed(seed)

		#Definimos los atributos de la clase EstrategiaParticionado
		self.nombreEstrategia="ValidacionCruzada"
		self.numeroParticiones=0
		self.particiones=[]

		#Creamos una lista de índices de 0 a n_lineas, y los desordenamos aleatoriamente
		lista=list(range(datos.n_lineas))
		random.shuffle(lista)

		#Separamos la lista de índices en k partes y elegimos una de ellas para test en cada vuelta del bucle
		for i in range(k):
			listas=[lista[j::k] for j in range(k)]
			aux=listas.pop(i)
			part=Particion()
			#Unificamos la lista de arrays en una sola lista
			part.indicesTrain=[y for x in listas for y in x]
			part.indicesTest.append(aux)
			self.particiones.append(part)
			self.numeroParticiones+=1

		return self.particiones
	   
   
    
#####################################################################################################

class ValidacionBootstrap(EstrategiaParticionado):

  # Crea particiones segun el metodo de boostrap
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
	def creaParticiones(self,datos,seed=None):

		random.seed(seed)

		#Definimos los atributos de la clase EstrategiaParticionado
		self.nombreEstrategia="ValidacionBootstrap"
		self.numeroParticiones=1
		self.particiones=[]

		#Creamos una particion
		particion = Particion()

		#Elegimos indices de filas al azar igual al numero de filas totales
		nsamples = datos.n_lineas
		while nsamples > 0:
			particion.indicesTrain=particion.indicesTrain + [random.randint(0,datos.n_lineas-1)]
			nsamples -= 1
		particion.indicesTrain=list(set(particion.indicesTrain))
		random.shuffle(particion.indicesTrain)

		#Recorremos todos los indices de filas y guardamos en indicesTest los que no fueron elegidos en indicesTrain
		for i in range(datos.n_lineas):
			if i not in particion.indicesTrain:
				particion.indicesTest=particion.indicesTest + [i]
		
		random.shuffle(particion.indicesTest)

		#Guardamos la particion en la clase EstrategiaParticionado
		self.particiones += [particion]

		return self.particiones
