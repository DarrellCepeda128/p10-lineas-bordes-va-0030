# ===============================
# Cepeda Darrell NC = 0030
# ===============================

import cv2
import numpy as np

# 1. Cargar la imagen asignada
nombre_imagen = 'orangutan_bordes.jpg'
img = cv2.imread(nombre_imagen)

if img is None:
    raise FileNotFoundError(f"No se encontró la imagen '{nombre_imagen}'. Revisa la ubicación del archivo.")

# Copia para dibujar resultados
img_harris = img.copy()

# 2. Convertir a escala de grises
gray = cv2.cvtColor(img_harris, cv2.COLOR_BGR2GRAY)

# 3. Convertir a tipo float32 (requerido por cv2.cornerHarris)
gray_float = np.float32(gray)

# 4. Aplicar el detector de esquinas de Harris
# Parametros:
# - src: Imagen de entrada en escala de grises (float32)
# - blockSize: Tamaño del vecindario considerado para la detección (ej. 2)
# - ksize: Apertura del operador Sobel utilizado (ej. 3)
# - k: Parámetro libre de la ecuación de Harris (típicamente entre 0.04 y 0.06)
blockSize = 2
ksize = 3
k = 0.04

dst = cv2.cornerHarris(gray_float, blockSize, ksize, k)

# 5. Dilatar los puntos detectados para que sean más visibles
dst_dilated = cv2.dilate(dst, None)

# 6. Definir umbral (threshold) y marcar las esquinas en rojo (BGR: [0, 0, 255])
umbral = 0.01 * dst_dilated.max()
img_harris[dst_dilated > umbral] = [0, 0, 255]

# 7. Mostrar resultados en ventana
cv2.imshow('Imagen Original - NC: 0030', img)
cv2.imshow('Deteccion de Esquinas Harris (Base) - orangutan_bordes.jpg', img_harris)

# Guardar resultado
cv2.imwrite('resultado_orangutan_base_0030.jpg', img_harris)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Programa realizado por Cepeda Darrell Nc = 0030")