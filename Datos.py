# coding=utf-8
import numpy as np
#np.set_printoptions(threshold=np.nan)

class Datos(object):
	
	

	# TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
	def __init__(self, nombreFichero):
		with open(nombreFichero,'r') as f:
			lines = f.readlines()
			
		self.TiposDeAtributos=('Continuo','Nominal')
		self.nominalAtributos=[]
		self.datos=[]
		self.diccionarios = []
		self.n_lineas = int(lines[0])
		self.nombreAtributos = lines[1].rstrip().split(",")
		self.tipoAtributos = lines[2].rstrip().split(",")
		
		#Comprobamos que los tipos de atributos son los soportados por el programa
		#Si están soportados, añadimos True a nominalAtributos y false en caso contrario
		for item in self.tipoAtributos: 
			if item==self.TiposDeAtributos[0]:
				self.nominalAtributos.append(False)
			elif item==self.TiposDeAtributos[1]:
				self.nominalAtributos.append(True)
			else:
				print("Error en el fichero de entrada. Tipo de atributos desconocido")
				return 
			
		#Descomponemos los datos en una lista de listas
		for x in lines[3:]:
			data = []
			for i in range(len(self.tipoAtributos)):
				data.append(x.rstrip().split(',')[i])
			self.datos.append(data)

		#Creamos conjuntos con todos los posibles valores para cada atributo

		conjuntos=[]
		for i in range(len(self.tipoAtributos)):
			dicc=set()
			for j in range(int(self.n_lineas)):
				dicc.add(self.datos[j][i])
					
			conjuntos.append(sorted(dicc))

		#Creamos a partir de los conjuntos los diccionarios para cada atributo nominal
		for i, conj in enumerate(conjuntos):
			dicc = {}
			if self.nominalAtributos[i] == True:
				for i, key in enumerate(conj):
					dicc.update({key:i})
			self.diccionarios.append(dicc)

		#Sustituimos los valores de cada atributo nominal por el valor guardado en el diccionario
		for dato in self.datos:
			for i, diccionario in enumerate(self.diccionarios):
				for item in diccionario.items():
					if item[0] == dato[i]:
						dato[i]=item[1]
				dato[i]=float(dato[i])
		#Creamos una matriz de Numpy con los datos generados anteriormente
		self.datos = np.array(self.datos)

	# TODO: implementar en la prctica 1
	def extraeDatos(self,idx):
		return self.datos[idx]






 
