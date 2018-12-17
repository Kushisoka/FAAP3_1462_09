from Clasificador import Clasificador
from math import *
from time import time
import numpy as np
from random import randint
from random import uniform
from random import shuffle

class ClasificadorAG(Clasificador):

	def __init__(self, tam_poblacion = 100, generaciones = 100, reglas_iniciales = 3, binaria = False, pc = 0.8, pz = 0.9, ps = 0.75):
		self.reglas = []
		self.tablas = []
		self.tam_poblacion = tam_poblacion
		self.generaciones = generaciones
		self.reglas_iniciales = reglas_iniciales
		self.binaria = binaria
		self.pc = pc
		self.pz = pz
		self.ps = ps
		self.mejor_fitness = []
		self.promedio_fitness = []


#######################################################################################################################################################################

#######		ENTRENAMIENTO	#################################################################################################################################

#######################################################################################################################################################################


	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

		#K = 1 + 3.322 log10 N
		self.k = int(1 +  3.322 * log10(len(datostrain)))

		datos_atributos = np.transpose(datostrain)[:-1]
		datos_clase = np.transpose(datostrain)[-1]

		self.num_atributos = len(datos_atributos)

		self.pm = 1/self.num_atributos

		for columna in datos_atributos:

			maxcolumn = np.max(columna)
			mincolumn = np.min(columna)

			A = (maxcolumn-mincolumn)/self.k

			tabla = []
			aux = []
			acc = mincolumn

			for idx in range(self.k):
				if idx != (self.k-1):
					aux = [idx+1] + [acc] + [acc + A]
					tabla += [aux]
					acc += A
				else:
					aux = [idx+1] + [acc] + [maxcolumn]
					tabla += [aux]
					acc += A

			self.tablas += [tabla]


		if self.binaria:

			# Poblar

			self.cromosomas = [[[ self.generar_bit() for _ in range(self.num_atributos*self.k)] + [randint(0,1)] for _ in range(randint(1, self.reglas_iniciales))]for _ in range(self.tam_poblacion)]

			return self.algotimo_genetico_binario(self.generaciones, datostrain)

		else:

			# Poblar

			self.cromosomas = [[[self.generar_gen() for _ in range(self.num_atributos)] + [randint(0,1)] for _ in range(randint(1, self.reglas_iniciales))]for _ in range(self.tam_poblacion)]

			return self.algotimo_genetico_entero(self.generaciones, datostrain)



	def algotimo_genetico_entero(self, generaciones, datostrain):

		if generaciones == 0:
			return

		start_time = time()

		# Evaluar

		scores = [self.score_entera(cromosoma, self.condiciones_excluyentes(datostrain))**2 for cromosoma in self.cromosomas]

		print(scores)

		# Seleccionar

		fitness = np.asarray(scores)
		denominador = np.sum(fitness)
		counts = 0
		seleccionados = []

		while counts < self.tam_poblacion:
			index = randint(0, len(scores) - 1)
			umbral = uniform(0, 1)
			if umbral < (scores[index]/denominador):
				seleccionados += [self.cromosomas[index]]
				counts += 1

		# Cruzar

		cruzados = []

		for seleccionado in seleccionados:
			umbral = uniform(0, 1)
			if umbral < self.pc:
				cruzado = self.cruzar_punto(seleccionado, seleccionados, self.condiciones_excluyentes(datostrain), randint(1, self.num_atributos - 1))
				if self.score_entera(seleccionado, self.condiciones_excluyentes(datostrain)) < self.score_entera(cruzado, self.condiciones_excluyentes(datostrain)):
					cruzados += [cruzado]

		reglas = []
		for seleccionado in seleccionados:
			reglas += [regla for regla in seleccionado]

		shuffle(reglas)

		recombinados = []
		i = 0
		while i < len(reglas)-1:
			n_group = randint(1, 3)
			while (n_group + i) > len(reglas)-1:
				n_group = randint(1, 3)
			recombinados += [reglas[i:i+n_group]]
			i += n_group

		# Mutar

		mutados = []

		for seleccionado in seleccionados:
			mutado = self.mutar_entera(seleccionado)
			if self.score_entera(seleccionado, self.condiciones_excluyentes(datostrain)) <= self.score_entera(mutado, self.condiciones_excluyentes(datostrain)):
				mutados += [mutado]

		# Repetir

		merged = seleccionados + cruzados + recombinados + mutados
		poblacion_final = [(self.score_entera(cromosoma, self.condiciones_excluyentes(datostrain)), cromosoma) for cromosoma in merged]
		poblacion_final.sort(reverse = True)

		cromosomas_seleccionados = int(self.tam_poblacion*self.ps)
		cromosomas_no_seleccionados = self.tam_poblacion - cromosomas_seleccionados
		self.cromosomas = [cromosoma for score, cromosoma in poblacion_final[:cromosomas_seleccionados]]

		self.cromosomas += [[[self.generar_gen() for _ in range(self.num_atributos)] + [randint(0,1)] for _ in range(randint(1, self.reglas_iniciales))]for _ in range(cromosomas_no_seleccionados)]


		scores_finales = [score for score, cromosoma in poblacion_final[:self.tam_poblacion]]
		scores_finales = np.asarray(scores_finales)

		self.mejor_fitness += [self.fitness(scores_finales[0])]
		self.promedio_fitness += [self.fitness(np.mean(scores_finales))]

		print("Generacion " + str(self.generaciones - generaciones + 1))

		print("Time Run:", str(time() - start_time))

		print("Score: " + str(poblacion_final[0][0]*100/len(datostrain)) +"%")

		return self.algotimo_genetico_entero(generaciones - 1, datostrain)

	def algotimo_genetico_binario(self, generaciones, datostrain):

		if generaciones == 0:
			return

		start_time = time()

		# Evaluar

		scores = [self.fitness(self.score_binaria(cromosoma, self.condiciones_no_excluyentes(datostrain))) for cromosoma in self.cromosomas]

		# Seleccionar

		fitness = np.asarray(scores)
		denominador = np.sum(fitness)
		counts = 0
		seleccionados = []

		while counts < self.tam_poblacion:
			index = randint(0, len(scores) - 1)
			umbral = uniform(0, 1)
			if umbral < (scores[index]/denominador):
				seleccionados += [self.cromosomas[index]]
				counts += 1

		# Cruzar

		cruzados = []

		for seleccionado in seleccionados:
			umbral = uniform(0, 1)
			if umbral < self.pc:
				cruzado = self.cruzar_punto(seleccionado, seleccionados, self.condiciones_no_excluyentes(datostrain), randint(1, self.num_atributos - 1), entera = False)
				if self.score_binaria(seleccionado, self.condiciones_no_excluyentes(datostrain)) < self.score_binaria(cruzado, self.condiciones_no_excluyentes(datostrain)):
					cruzados += [cruzado]

		reglas = []
		for seleccionado in seleccionados:
			reglas += [regla for regla in seleccionado]

		shuffle(reglas)

		recombinados = []
		i = 0
		while i < len(reglas)-1:
			n_group = randint(1, 3)
			while (n_group + i) > len(reglas)-1:
				n_group = randint(1, 3)
			recombinados += [reglas[i:i+n_group]]
			i += n_group

		# Mutar

		mutados = []

		for seleccionado in seleccionados:
			mutado = self.mutar_entera(seleccionado)
			if self.score_binaria(seleccionado, self.condiciones_no_excluyentes(datostrain)) <= self.score_binaria(mutado, self.condiciones_no_excluyentes(datostrain)):
				mutados += [mutado]

		# Repetir

		merged = seleccionados + cruzados + recombinados + mutados
		poblacion_final = [(self.score_binaria(cromosoma, self.condiciones_no_excluyentes(datostrain)), cromosoma) for cromosoma in merged]
		poblacion_final.sort(reverse = True)

		cromosomas_seleccionados = int(self.tam_poblacion*self.ps)
		cromosomas_no_seleccionados = self.tam_poblacion - cromosomas_seleccionados
		self.cromosomas = [cromosoma for score, cromosoma in poblacion_final[:cromosomas_seleccionados]]

		self.cromosomas += [[[self.generar_bit() for _ in range(self.num_atributos*self.k)] + [randint(0,1)] for _ in range(randint(1, self.reglas_iniciales))]for _ in range(cromosomas_no_seleccionados)]


		scores_finales = [score for score, cromosoma in poblacion_final[:self.tam_poblacion]]
		scores_finales = np.asarray(scores_finales)

		self.mejor_fitness += [self.fitness(scores_finales[0])]
		self.promedio_fitness += [self.fitness(np.mean(scores_finales))]

		print("Generacion " + str(self.generaciones - generaciones + 1))

		print("Time Run:", str(time() - start_time))

		print("Score: " + str(poblacion_final[0][0]*100/len(datostrain)) +"%")

		return self.algotimo_genetico_binario(generaciones - 1, datostrain)

#######################################################################################################################################################################

#######		CLASIFICA		#################################################################################################################################

#######################################################################################################################################################################


	def clasifica(self, datostest, atributosDiscretos, diccionario):

		resultado = []
		for individuo in datostest:
			clases = np.array([])
			for regla in self.cromosomas[0]:
				if self.binaria:
					clase = self.predict_regla_binaria(regla, individuo)
					clases = np.append(clases, clase)
				else:
					clase = self.predict_regla_entera(regla, individuo)
					clases = np.append(clases, clase)

			freqs = np.unique(clases, return_counts=True)

			for i, counts in enumerate(freqs[1]):
				if counts == np.max(freqs[1]):
					pos = i

			resultado += [freqs[0][pos]]

		return resultado

#######################################################################################################################################################################

#######		FUNCIONES UTILES		#################################################################################################################################

#######################################################################################################################################################################

	###############################################################################
	#
	#	fitness: Devuelve el fitness de una puntuacion
	#
	###############################################################################


	def fitness(self, score):

		return score**2

	###############################################################################
	#
	#	generar_gen: Devuelve un numero aleatorio entre 0 y k con una probabilidad de 0s en funcion de pz
	#
	###############################################################################


	def generar_gen(self):

		umbral = uniform(0,1)

		if umbral < self.pz:
			return 0
		else:
			return randint(0, self.k)

	###############################################################################
	#
	#	generar_bit: Devuelve 1 o 0 en funcion de pz
	#
	###############################################################################

	def generar_bit(self):

		umbral = uniform(0,1)

		if umbral < self.pz:
			return 0
		else:
			return 1


	###############################################################################
	#
	#	condiciones_excluyentes: datos
	#
	###############################################################################

	def condiciones_excluyentes(self, datos):

		condiciones = []

		for individuo in datos:
			i=0
			aux=[]
			for atributo in individuo[:-1]:
				for intervalo in self.tablas[i]:
					if (intervalo[1] <= atributo) and (atributo <= intervalo[2]):
						aux += [intervalo[0]]
						break
				i += 1
			aux += [int(individuo[-1])]
			condiciones += [aux]

		return condiciones


	###############################################################################
	#
	#	condiciones_no_excluyentes: datos
	#
	###############################################################################


	def condiciones_no_excluyentes(self, datos):

		condiciones = []

		for individuo in datos:
			i=0
			aux = []
			for atributo in individuo[:-1]:
				for intervalo in self.tablas[i]:
					if (intervalo[1] <= atributo) and (atributo <= intervalo[2]):
						aux += [1]
					else:
						aux += [0]
				i += 1
			aux += [int(individuo[-1])]
			condiciones += [aux]

		return condiciones



	###############################################################################
	#
	#	predict_regla_entera: regla, individuo_de_datos_excluyentes
	#
	###############################################################################

	def predict_regla_entera(self, regla, individuo):

		i = 0
		for atributo in regla[:-1]:
			if atributo != 0:
				if atributo != individuo[i]:
					return regla[-1] ^ 1
			i += 1

		return regla[-1]

	###############################################################################
	#
	#	predict_regla_binaria: regla, individuo_de_datos_no_excluyentes
	#
	###############################################################################


	def predict_regla_binaria(self, regla, individuo):

		for i in range(self.num_atributos):
			flag = 0
			acc = 0
			for j in range(self.k):
				if regla[i*self.k+j] == 1:
					flag = 1
					acc = individuo[i*self.k+j] or acc
					if acc == 1:
						break
			if acc == 0 and flag == 1:
				return regla[-1] ^ 1

		return regla[-1]


	###############################################################################
	#
	#	score_entera: cromosoma, datos_excluyentes
	#
	###############################################################################

	def score_entera(self, cromosoma, datos):

		score = 0.0

		for individuo in datos:
			clases = np.array([])
			for regla in cromosoma:
				clase = self.predict_regla_entera(regla, individuo)
				clases = np.append(clases, clase)

			freqs = np.unique(clases, return_counts=True)

			for i, counts in enumerate(freqs[1]):
				if counts == np.max(freqs[1]):
					pos = i

			if freqs[0][pos] == individuo[-1]:
				score += 1
			else:
				score += 0

		return score

	###############################################################################
	#
	#	score_binaria: cromosoma, datos_no_excluyentes
	#
	###############################################################################

	def score_binaria(self, cromosoma, datos):

		score = 0.0

		for individuo in datos:
			clases = np.array([])
			for regla in cromosoma:
				clase = self.predict_regla_binaria(regla, individuo)
				clases = np.append(clases, clase)

			freqs = np.unique(clases, return_counts=True)

			for i, counts in enumerate(freqs[1]):
				if counts == np.max(freqs[1]):
					pos = i

			if freqs[0][pos] == individuo[-1]:
				score += 1
			else:
				score += 0

		return score

	def cruzar_punto(self, seleccionado, seleccionados, datos, punto, entera = True):

		scores = []
		n_reglas = len(seleccionado)

		cromosoma = seleccionados[randint(0, len(seleccionados)-1)]

		for regla in seleccionado:
			for regla2 in cromosoma:
				aux1 = regla[:punto] + regla2[punto:]
				aux2 = regla2[:punto] + regla[punto:]
				if entera:
					score = self.score_entera([aux1], datos)
					score2 = self.score_entera([aux2], datos)
				else:
					score = self.score_binaria([aux1], datos)
					score2 = self.score_binaria([aux2], datos)

				if score < score2:
					scores += [(score2, aux2)]
				else:
					scores += [(score, aux1)]

		resultado = []
		for _ in range(n_reglas):
			score, regla = max((score, regla) for score, regla in scores)
			resultado += [regla]
			scores.remove((score, regla))

		return resultado


	###############################################################################
	#
	#	mutar_entera: cromosoma_entero
	#
	###############################################################################

	def mutar_entera(self, seleccionado):

		mutado = []

		for regla in seleccionado:
			regla_mutada = []
			for gen in regla[:-1]:
				umbral = uniform(0, 1)
				if umbral < self.pm:
					gen_mutado = randint(0, self.k)
					while gen == gen_mutado:
						gen_mutado = randint(0, self.k)
					gen = gen_mutado
				regla_mutada += [gen]
			regla_mutada += [regla[-1]]
			mutado = [regla_mutada]

		return mutado

	###############################################################################
	#
	#	mutar_binaria: cromosoma_binario
	#
	###############################################################################

	def mutar_binaria(self, seleccionado):

		mutado = []

		for regla in seleccionado:
			regla_mutada = []
			for gen in regla[:-1]:
				umbral = uniform(0, 1)
				if umbral < self.pm:
					gen = gen ^ 1
				regla_mutada += [gen]
			regla_mutada += [regla[-1]]
			mutado = [regla_mutada]

		return mutado


#######################################################################################################################################################################

#######		Funciones que ya no se usan		#################################################################################################################################

#######################################################################################################################################################################

	def predict_binaria(self, genotipo, cromosomas):
		
		score = 0.0
		for cromosoma in cromosomas:
			if self.predict(genotipo, cromosoma) == cromosoma[-1]:
				score += 1
			else:
				score += 0

		score /= len(cromosomas)

		return score

	def predict(self, genotipo, cromosoma):
		if not isinstance(genotipo, int):
			if 'or' in genotipo.keys():
				resultado = 0
				for expresion in genotipo['or']:
					resultado = self.predict(expresion,cromosoma) or resultado
					print(str(resultado) + " or ")
			elif 'and' in genotipo.keys():
				resultado = 1
				for expresion in genotipo['and']:
					resultado = self.predict(expresion,cromosoma) and resultado
					print(str(resultado) + " and ")
		else:
			return cromosoma[genotipo]

		if 'clase' in genotipo.keys():
			print("Resultado Final: " + str(resultado))
			if resultado == 1:
				return genotipo['clase']
			else:
				return genotipo['clase'] ^ 1
		else:
			print("resultado: " + str(resultado))
			return resultado

	# Genera los genotipos inicales del algoritmo mejorado
	def generar_cromosomas(self, datos):

		cromosomas = []

		for individuo in datos:
			i=0
			cromosoma = []
			for atributo in individuo[:-1]:
				for intervalo in self.tablas[i]:
					if (intervalo[1] <= atributo) and (atributo <= intervalo[2]):
						cromosoma += [1]
					else:
						cromosoma += [0]
				i += 1
			cromosoma += [individuo[-1]]
			cromosomas += [cromosoma]

		return cromosomas

	# Genera los genotipos inicales del algoritmo mejorado
	def generar_genotipos(self, cromosomas):

		genotipos = []

		for cromosoma in cromosomas:

			iterador = 0
			genotipo = {}
			gen_atributo = []

			for tabla in self.tablas:
				aux = {}
				ors = []
				for intervalo in tabla:
					ors += [iterador]
					iterador += 1

				aux['or'] = ors
				gen_atributo += [aux]

			genotipo['and'] = gen_atributo
			genotipo['clase'] = cromosoma[-1]
			genotipos += [genotipo]

		return genotipos