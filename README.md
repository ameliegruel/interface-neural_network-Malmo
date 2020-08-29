# Développement d'un réseau neuronal adapté à la réalisation d'une tâche de survie

Projet de 6 mois en Python 3  
Implémentation du réseau de neurones développé par Ardin et al. (*PLOS Computational Biology*, 2016) à l'aide de la bibliothèque **BindsNET**  
Implémentation de l'interface sensori-motrice et du monde virtuel avec la plateforme **Malmo**  

## 1. Structure du projet

## 2. Installation de Malmo

### a) Installation de Malmo sur Windows 10

Les étapes nécessaires à l’installation du logiciel Malmo à partir de la version *pre-built* sur un environnement Windows 10 professionnel sont présentées ci-dessous. 

Il faut d’abord s’assurer de la présence sur la machine de : 
* Python 3.6 
* le logiciel **Java JDK 8**
* les fonctionnalités gym, lxml, numpy, pillow et future, installables à l’aide de la commande ``pip3 install gym lxml numpy pillow future``. 

Afin que Malmo fonctionne, il est possible que la réinstallation de Java OpenJDK 8 sur la machine soit nécessaire. Pour cela, une version adaptée aux paramètres de la machine est téléchargeable à partir du site *JavaDownloads*. Une fois la version téléchargée, il faut s'assurer que la variable d'environnement **JAVA\_HOME** pointe bien vers cette version et non l'ancienne.
Une fois ces étapes préliminaires accomplies, l’installation de Malmo se déroule ainsi : 
1. Il faut d’abord télécharger la version la plus récente (pour Windows et Python 3.6) du *pre-built*, sur ce lien : https://github.com/Microsoft/malmo/releases. 
1. Le ﬁchier .zip téléchargé doit être décompressé, puis renommer et déplacé à l'emplacement souhaité. 
1. La variable d'environnement **MALMO\_XSD\_PATH** doit être indiquée à la machine, c’est-à-dire le chemin d’accès menant au dossier "Schémas" contenu dans le dossier général "Malmo". Pour ce faire, il faut créer une nouvelle variable d'environnement système nommée "MALMO\_XSD\_PATH", à laquelle on attribue la valeur **C:\path\Malmo\Schemas** (en remplaçant path par le bon chemin d’accès vers le dossier "Malmo"). 

Il est désormais possible de tester le bon fonctionnement de Malmo. Le github dédié à Malmo propose de lancer Minecraft en passant par le mod installé, puis de tester un exemple d’agent. Nous essayerons un agent Python car il a été décidé préalablement d’utiliser ce langage, mais des exemples d’agents codés en de nombreux langages diﬀérents (C++, C\#, Java, Python) sont fournis par Malmo. 

La commandes suivantes permettent de lancer Minecraft en utilisant le mod installé et en partant du dossier "Malmo" :  
``cd Minecraft/``  
``launchClient -port 10000``  
La dernière ligne affichée par la terminal indiquera "Building 95\%" : ne vous inquiétez pas et n’attendez pas 100%, tant que la fenêtre de jeu Minecraft est bien lancée. De plus, le Mod choisit le port 10000 par défaut, et navigue de port en port jusque 11000 aﬁn de trouver un port libre. 

Une fois la fenêtre Minecraft lancée, il faut ouvrir un nouveau terminal et entrer les commandes suivantes :  
``cd Python_Examples``  
``python3 run_mission.py``  
Une fois cet exemple réalisé, on peut suivre des tutoriels fournis par Malmo et expliqué sur le document suivant : https//github.com/Microsoft/malmo/blob/master/Malmo/samples/Python\_examples/Tutorial.pdf
