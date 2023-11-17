# Importação das bibliotecas necessárias
import cv2
import numpy as np

# Inicialização da captura de vídeo da webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Inicializa a captura de vídeo da webcam (0 indica a câmera padrão)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Define a largura do quadro de captura como 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Define a altura do quadro de captura como 720 pixels

# Carregamento do classificador de face (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialização das variáveis
target_size = 100  # Tamanho do quadrado ao redor do rosto

# Loop principal
while True:
    # Captura de vídeo
    success, img = cap.read()  # Lê um frame do vídeo
    img = cv2.flip(img, 1)  # Inverte horizontalmente a imagem para evitar espelhamento

    # Conversão para escala de cinza para o classificador de face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza

    # Detecção de faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Detecta faces na imagem

    # Desenho do quadrado ao redor do rosto
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Desenha um retângulo preto ao redor do rosto
        target_x = x + w // 2 - target_size // 2  # Calcula a coordenada x do canto superior esquerdo do quadrado alvo
        target_y = y + h // 2 - target_size // 2  # Calcula a coordenada y do canto superior esquerdo do quadrado alvo
        cv2.rectangle(img, (target_x, target_y), (target_x + target_size, target_y + target_size), (0, 0, 255), 2) # Desenha um quadrado vermelho alvo no centro do rosto

    # Exibição da imagem
    cv2.imshow("Image", img)  # Exibe a imagem

    # Verificação de tecla para sair
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exit")
        break
    elif key == 27:  # Verifica se a tecla pressionada é a tecla 'Esc'
        print("Exit (Esc pressed)")
        break

# Liberação da captura de vídeo e fechamento da janela
cap.release()  # Libera os recursos da câmera
cv2.destroyAllWindows()  # Fecha a janela
