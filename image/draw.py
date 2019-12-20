########## INFORMATIONS ####################################
#Author:     Sherlon Almeida da Silva                      #
#University: University of Sao Paulo                       #
#Country:    Brazil                                        #
#Created:    06/18/2019                                    #
############################################################

###### Useful Libraries ####################################
import matplotlib.pyplot as plt                            #
import numpy as np                                         #
import imageio                                             #
import sys                                                 #
############################################################

"""Classe que define a figura e desenha as linhas para reducao de ruido ou realce de bordas"""
class Draw:
    """Funcao inicial da classe Draw"""
    def __init__(self, img):
        self.size = 8       #Tamanho do quadrado (2*size) em X e (2*size) em Y
        self.radius = 3     #Tamanho do Raio do circulo que sera pintado
        self.points = []    #Pontos que formam as linhas desenhadas
        self.img = np.array(img, copy=True).astype(np.float64)       #Variavel para armazenar a imagem
        self.last_img = np.array(img, copy=True).astype(np.float64)  #Variavel para armazenar a imagem uma acao antes (Para permitir desfazer acao)
        self.m,self.n = img.shape                                    #Obtem as dimensoes da imagem
        
        #Inicializa o Canvas
        '''
        self.canvas_update()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.get_position)  #Define os eventos de clique
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.get_key)          #Define os eventos de tecla
        '''
    #Atualizar a imagem original e o Canvas
    def canvas_update(self):
        plt.imshow(self.img, cmap="gray")
        self.fig = plt.gcf() # Obtem a figura atual
        self.ax = plt.gca()  # get axis handle
        self.ax.set_title('Draw lines to avoid the Noise\nLine_Size: %d' %(self.radius))
        self.fig.canvas.draw()
        
    """Funcao para aumentar/reduzir a espessura da linha"""
    """e Funcao para desfazer a ultima acao realizada"""
    def get_key(self, event):
        sys.stdout.flush()
        if event.key == '+':
            if self.radius < 10: self.radius += 1
            print('Aumentou(%c) a espessura da linha: %d' % (event.key, self.radius))
        elif event.key == '-':
            if self.radius > 1: self.radius -= 1
            print('Diminuiu(%c) a espessura da linha: %d' % (event.key, self.radius))
        elif event.key == 'backspace':
            print("A ultima modificacao foi desfeita!")
            self.img = np.array(self.last_image, copy=True).astype(np.float64)
        self.canvas_update()
    
    """Funcao para desenhar circulos nos pontos desejados - Auxilia na formacao de linhas mais espessas"""
    def draw_circle(self, x, y, color):
        if (x-self.size >= 0) and (x+self.size < self.m) and (y-self.size >= 0) and (y+self.size < self.n):
            for i in range(x-self.size, x+self.size, 1):
                for j in range(y-self.size, y+self.size, 1):
                    d = ((x-i)**2+(y-j)**2)**(1/2)
                    if d < self.radius:
                        self.img[i,j] = color
    
    """Funcao para obter as coordenadas dos pontos na imagem, para a seguir tracar uma reta"""
    def get_position(self, event):
        #Obtem as coordenadas na imagem (Obs.: Eh invertido, pois em proc imagens o plano cartesiano eh rotacionado em 90 graus em sentido horario)
        y,x = int(event.xdata), int(event.ydata)
        
        #Salva o estado atual da imagem em self.last_image
        self.last_image = np.array(self.img, copy=True).astype(np.float64)
        
        if len(self.points) < 1:    #Se ainda nao obteve um ponto
            self.points.append(x)       #Salva a coordenada X
            self.points.append(y)       #Salva a coordenada Y
        else:                       #Se ja obteve o primeiro ponto
            #Define a cor a ser atribuida a reta
            if event.button == 1:    #Botao Esquerdo do Mouse
                color = 0                #Pinta de Preto
            elif event.button == 3:  #Botao Direito do Mouse
                color = 1                #Pinta de Branco
            
            #Define os pontos
            x1,y1 = self.points      #Pontos P1
            x2,y2 = x,y              #Pontos P2
            if (x2-x1) != 0: #Caso nao seja uma linha horizontal (Calcula normalmente a equacao da reta dados 2 pontos P1 e P2)
                m = (y2-y1)/(x2-x1)                 #Calcula o coeficiente angular
                interval = np.linspace(x1, x2, 100) #Discretiza 100 pontos na reta
                for X in interval:                  #Percorre os pontos em X
                    Y = m*(X-x1)+y1                         #Calcula o valor de Y, dada a equacao da reta obtida
                    self.draw_circle(int(X), int(Y), color) #Desenha o ponto na imagem
            else: #Caso seja uma linha horizontal (Tratar a divisao por zero)
                interval = np.linspace(y1,y2,100) #Discretiza o eixo Y
                for Y in interval:                #Percorre os pontos em Y com o X fixo, pois eh uma linha reta horizontal
                    self.draw_circle(int(x1), int(Y), color)#Desenha o ponto na imagem
            
            self.canvas_update() #Atualiza o Canvas
            self.points = []     #Esvazia o ponto atual

