# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from EstrategiaParticionado import Particion
from EstrategiaParticionado import ValidacionCruzada
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionBootstrap
from ClasificadorAG import ClasificadorAG


path1="data/wdbc.data"
path2="data/tic-tac-toe.data"

dataset=  Datos(path1)
dataset2=  Datos(path2)


estrategia=ValidacionSimple()
estrategia.creaParticiones(dataset, 0.8)

estrategia2=ValidacionSimple()
estrategia2.creaParticiones(dataset2, 0.8)

#clasificador=ClasificadorAG(reglas_iniciales=1)

clasificador=ClasificadorAG(binaria = False, reglas_iniciales=1)

"""
ii = estrategia.particiones[-1].indicesTrain
plotModel(dataset_wdbc.datos[ii, 0], dataset_wdbc.datos[ii, 1], dataset_wdbc.datos[ii, -1] != 0, clasificador, "RL LR=1;EPOC=10", dataset_wdbc.diccionarios)
print(clasificador.validacion(estrategia, dataset, clasificador))
print(clasificador.validacion(estrategia2, dataset2, clasificador))"""

clasificador.entrenamiento(dataset.extraeDatos(estrategia.particiones[0].indicesTrain),dataset.nominalAtributos,dataset.diccionarios)


