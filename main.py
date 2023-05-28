import pygame as pg
import random
import neat
import os
import Assets as ass
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
pg.init()
pg.font.init()
FONT_POINTS = pg.font.SysFont('arial', 100)
g = 0
gerations = 10000
best_fitness = []
current_fitness = []
fitness_mid = []

nets = []
cars = []
ge = []
def rodar(caminho_config):
    # Configurando NEAT
    config = neat.config.Config(neat.DefaultGenome,
                               neat.DefaultReproduction,
                               neat.DefaultSpeciesSet,
                               neat.DefaultStagnation,
                               caminho_config)
    global populacao
    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    populacao.add_reporter(neat.StatisticsReporter())
    populacao.run(game, gerations)
    


    
# Função de jogo 
def game(genomes, config):
    def car_IA():
        return [122, 570]

    # variaveis da rede neural
    global g
    g += 1
    
    
    
    
    # criando a janela
    window = pg.display.set_mode((1024, 768))
    pg.display.set_caption('IA Aprende Dirigir')
     
    # variaveis do jogo
    points = 0
    velocity = 90
    largura_cars = 100
    comprimento_cars =200
    car_pos_center = 122
    car_distance_center = random.randint(-3500, -500)
    car_pos = random.choice([122,235])
    car_distance = -200
    car_left = random.choice(ass.sprites.carros)
    car_center = random.choice(ass.sprites.carros)
    car_right = random.choice(ass.sprites.carros)
    car_sprite = random.choice(ass.sprites.carros)
    car_sprite_center = random.choice(ass.sprites.carros)
    
    running = True
    fps = pg.time.Clock()

    # listando os genomas
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(car_IA())
        genome.fitness = 0
        ge.append(genome)
    def distance_y(dis):
        distance = ((dis + 200) - 570 )
        return distance
    def distance_x(Ia_pos, cars_pos):
        distance = (Ia_pos - cars_pos)
        return distance
    # função para criar os carros
    def spaw_cars_lat(window):
        return window.blit(car_sprite, (car_pos, car_distance))
    def spaw_cars_center(window):
        return window.blit(car_sprite_center, (car_pos_center, car_distance_center))
    # função para desenhar os graficos
    def draw_graph(window, x, y):     
        plt.close()
        fig = plt.figure(figsize=(7, 4))
        plot = fig.add_subplot(1, 1, 1)
        ger = range(len(best_fitness))
        plt.plot(ger, fitness_mid, label='Fitness Medio')
        plt.plot(ger, best_fitness, label='Melhor Fitness')
        plt.title('Evolução da Aptidão das Gerações')
        plt.ylabel('Fitness')
        plt.xlabel('Gerações')
        plt.legend()
        

        canvas = FigureCanvas(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()

        graph_surface = pg.image.fromstring(raw_data, size, 'RGB')

        return window.blit(graph_surface, (x, y))
    
    # Loop principal do jogo...
    while running:
        kd = True
        pg.display.flip()
        # definindo valor delta ,passagem de tempo linear
        fps.tick(120)
        # verificando se algum evento no pygame foi do tipo quit()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                pg.quit()
                quit()
            
        # tratamento para caso ocorra alguma excessão
        try:
            # loop de interação da rede neural
            for i, car in enumerate(cars):
                
                # adicionando dados nas listas para o grafico
                current_fitness.append(ge[0].fitness)
                if i >= 0:
                    ge[i].fitness += 0.1 
                    # ativando a rede neural e passando os inputs
                    output = nets[i].activate((car[0],
                                               velocity,
                                               30,
                                               largura_cars,
                                               comprimento_cars,
                                               car_pos_center,
                                               car_pos,
                                               distance_y(car_distance_center),
                                               distance_y(car_distance),distance_x(car[0], car_pos),distance_x(car[0],car_pos_center),
                                               car_pos_center,
                                               car[1],
                                               ))
                    

                    # movimentando os carro da IA e limitando area da pista
                    if (output[0] > 0.5) and ((car[0] >= 2) and (car[0] <= 237)):
                        car[0] += 28
                            
                    
                    elif (output[1] > 0.5)and((car[0] >= 2) and (car[0] <= 237)):
                        car[0] -= 28
                    elif (output[2] > 0.5):
                        car[0] = car[0]
            for i, car in enumerate(cars): 
                if i >= 0:
                    # verificando se ouve colisão 
                    if (car[0] <= 105)and(car_pos == 5) and (car_distance >= 370):
                        car_sprite = ass.sprites.carro_queimado
                        ge[i].fitness -= 5
                        ge.pop(i)
                        cars.pop(i)
                        nets.pop(i)
                        
                    if ((car[0] >= 22) and (car[0] <= 222)) and (car_distance_center >= 370):
                        car_sprite_center = ass.sprites.carro_queimado
                        ge[i].fitness -= 5
                        ge.pop(i)
                        cars.pop(i)
                        nets.pop(i)
                    if ((car[0] >= 135) and (car_pos == 235)) and (car_distance >= 370):
                        car_sprite = ass.sprites.carro_queimado
                        ge[i].fitness -= 5
                        ge.pop(i)
                        cars.pop(i)
                        nets.pop(i)
                    if (car[0] < 0 or car[0] > 240):
                        ge[i].fitness -= 5
                        ge.pop(i)
                        cars.pop(i)
                        nets.pop(i)
                        

                    # checando se fez uma ultrapassagem e dando pontos para rede
                    if (car_distance > 770):
                        #ge[i].fitness += 0.2 
                        car_sprite = random.choice(ass.sprites.carros)
                        car_distance = -200
                        car_pos = random.choice([5, 235])
                        points += 1
                    if (car_distance_center > 770):
                        #ge[i].fitness += 0.2
                        car_sprite_center = random.choice(ass.sprites.carros)
                        car_distance_center = random.randint(-3500 , -500)
                        car_pos_center = 122
                        points += 1
                    if (car[0] > 105) and (car_pos ==5) and (car_distance >=370):
                        ge[i].fitness += 0.2
                    if (car[0] < 135) and (car_pos ==235) and (car_distance >= 370):
                        ge[i].fitness += 0.2
                    window.blit(ass.sprites.ia_car, car)
                    pg.display.update()  

                    
                    
        except:
            fitness_mid.append(sum(current_fitness)/ len(current_fitness))
            best_fitness.append(max(current_fitness))
            running = False
            break
        # caso todos os carros batam comeca uma nova geracão
        if len(ge) == 0:
            fitness_mid.append(sum(current_fitness)/ len(current_fitness))
            best_fitness.append(max(current_fitness))
            running = False
            break    

        # mostrando tudo na janela
        points_text = FONT_POINTS.render('Pontos: ' + str(points), 1, (255, 255, 255))
        geration_text = FONT_POINTS.render('Geração: ' + str(g), 1, (255, 255, 255))
        car_text = FONT_POINTS.render('Carros: ' + str(len(cars)), 1, (255, 255, 255))
        window.blit(ass.sprites.pista, (0,0))
        spaw_cars_lat(window)
        spaw_cars_center(window)
        window.blit(points_text, (353, 10))
        window.blit(car_text, (353, 100))
        window.blit(geration_text, (353, 200))
        draw_graph(window, 350, 350)
        car_distance += velocity 
        car_distance_center += velocity 
        
        
        
        
        
    

if __name__ == '__main__':
    # passando o caminho do arquivo de configuração do neat
    caminho = os.path.dirname(__file__)
    caminho_config = os.path.join(caminho, 'config.txt')
    rodar(caminho_config)