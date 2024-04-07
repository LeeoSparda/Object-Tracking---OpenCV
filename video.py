import cv2
import sys
from random import randint


#BOOSTING: Este é um algoritmo de rastreamento baseado em aprendizado de máquina. 
#Ele é adequado para cenas estáticas com iluminação uniforme, mas pode não funcionar bem em situações com mudanças drásticas na aparência do objeto.

#MIL (Multiple Instance Learning): Este algoritmo divide o objeto em várias "instâncias" e considera todas elas durante o rastreamento. 
#Ele é eficaz em cenas com oclusões parciais, mas pode ser mais lento em comparação com outros algoritmos.

#KCF (Kernelized Correlation Filters): O KCF é uma versão otimizada do algoritmo MOSSE. Ele usa kernels para calcular a similaridade entre a região rastreada e o modelo do objeto.
#O KCF é rápido e eficaz em cenas com movimentos rápidos e mudanças na aparência do objeto.

#TLD (Tracking, Learning and Detection): Este algoritmo combina rastreamento, aprendizado e detecção para lidar com oclusões temporárias e reencontrar objetos perdidos.
#Ele pode ser mais robusto em cenários desafiadores, mas também pode ser mais lento.

#MEDIANFLOW: Este é um método de rastreamento óptico baseado em fluxo de pixels.
#Ele é robusto em cenas com movimentos rápidos e mudanças na iluminação, mas pode não ser tão preciso em cenas com oclusões prolongadas.

#MOSSE (Minimum Output Sum of Squared Error): Este algoritmo é eficiente e adequado para rastreamento em tempo real. 
#Ele calcula um filtro discriminativo de correlação para rastrear o objeto, mas pode não ser tão preciso em comparação com outros métodos.

#CSRT (Channel and Spatial Reliability Tracker): O CSRT é uma versão mais avançada do KCF, que utiliza informações de canais de cor e confiabilidade espacial para melhorar a precisão do rastreamento. 
#Ele é rápido e preciso em várias condições, mas pode ser mais exigente em termos de recursos computacionais.

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture('D:/Desktop/CG/Atividade5/79177-568207010_tiny.mp4')
if not video.isOpened():
    print('Erro ao carregar o vídeo!')
    sys.exit()

frame_index = 14  # Inicializa o índice do quadro como 0

while True:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video.read()  

    if not ok:
        break

    bbox = cv2.selectROI(frame) 
    print(bbox)

    ok = tracker.init(frame, bbox)
    print(ok)

    colors = (randint(0, 255), randint(0,255), randint(0, 255)) # RGB -> BGR

    while True:
        ok, frame = video.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)
        if ok == True:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)
        else:
            cv2.putText(frame, 'Tracking failure!', (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255))

        cv2.putText(frame, tracker_type, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255))

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0XFF == 27: 
            break

    frame_index += 1

video.release()
cv2.destroyAllWindows()
