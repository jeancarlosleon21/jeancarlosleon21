import cv2
import random
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from yolov5MultiplePolygon import CountObject

#Creo los videos de entrada y salida para llamar la clase del script de python: yolov5MultiplePolygon.py
input_video = 'demo2.mp4'
output_video = 'resultado1.mp4'

#Verifica si el script está siendo ejecutado como un programa principal o si está siendo importado como un módulo en otro script.
if __name__ == "__main__":
    # Crear una instancia de CountObject y procesar el video
    obj = CountObject(input_video, output_video)
    obj.process_video()


# Para usar la clase CountObject2 utilizo el video procesado del anterior script utilizando la clase CountObject para despues añadir
# las etiquetas de los IDs por regiones de cada persona.
videopath = 'resultado1.mp4'
output_path = 'resultado2.mp4'

#defino una clase para despues poder importarla en el programa principal, pero en este caso en en el programa principal lo unico que utilizo
#es el programa detectos_personas_id4.py y corro el programa

class CountObject2:

    #inicializa una instancia de la clase. Recibe tres parámetros: self, que hace referencia a la instancia misma, videopath, que es la ruta del video de entrada, y output_path, que es la ruta donde se guardarán los resultados de la detección de objetos.
    def __init__(self, videopath, output_path):
        self.videopath = videopath #asigna la ruta del video de entrada proporcionada al atributo videopath de la instancia actual de la clase.
        self.output_path = output_path #asigna la ruta de salida proporcionada al atributo output_path de la instancia actual de la clase.
        self.model = YOLO("yolov5s.pt", task="detect") #crea una instancia del modelo YOLO (You Only Look Once) con el archivo de pesos "yolov5s.pt". El modelo se inicializa para la tarea de detección de objetos

    #defino la funcion para el detector; los resultados de la detección de objetos en la imagen proporcionada, y show_id, que es un indicador booleano opcional para mostrar o no la identificación de los objetos detectados.
    def draw_results(self, image, image_results, show_id=False):
        annotator = Annotator(image.copy()) #Se crea un objeto Annotator inicializado con una copia de la imagen origina
        #Se itera sobre cada objeto detectado en image_results.boxes
        for box in image_results.boxes:   
            b = box.xyxy[0]
            cls = int(box.cls)
            conf = float(box.conf)
            label = f"{self.model.names[cls]} {round(conf * 100, 2)}"
            if show_id:
                label += f' id:{int(box.id)}'
            if cls == 0 and conf >= 0.35:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                annotator.box_label(b, label, color=color)
        image_annotated = annotator.result() #Se obtiene la imagen anotada con las cajas delimitadoras y etiquetas agregadas.
        return image_annotated

    #Esta funcion me sirve para Abrir el video de entrada usando cv2.VideoCapture y obtiene información importante sobre el video, como el ancho y alto del fotograma, la velocidad de fotogramas (FPS) y el número total de fotogramas.
    #Crea un objeto cv2.VideoWriter para el video de salida utilizando la ruta proporcionada en self.output_path. Este objeto se utiliza para escribir los fotogramas procesados con los resultados de la detección.
    def process_video(self):
        video = cv2.VideoCapture(self.videopath)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        #Inicia un bucle mientras haya fotogramas disponibles en el video de entrada.

        #Lee un fotograma del video de entrada usando video.read().
        #Utiliza el método track del modelo para realizar el seguimiento de los objetos en el fotograma actual. Los parámetros especificados controlan la confianza mínima (conf), las clases de objetos a seguir (classes), el tipo de algoritmo de seguimiento (tracker), etc.
        #Llama al método draw_results para dibujar los resultados de la detección y seguimiento en el fotograma actual.
        #Escribe el fotograma procesado con los resultados en el video de salida utilizando output_video.write().
        #Actualiza el contador de fotogramas procesados y muestra un mensaje de progreso.
        while True:
            ret, frame = video.read()
            if not ret:
                break
            #utiliza el modelo para realizar el seguimiento de objetos en el fotograma actual del video, utilizando un umbral de confianza específico y un algoritmo de seguimiento definido. Los resultados del seguimiento se almacenan en la variable results_track.        
            results_track = self.model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
            image_with_results = self.draw_results(frame, results_track[0], show_id=True)

            output_video.write(image_with_results)

            frame_count += 1
            print(f"Procesando fotograma {frame_count}/{num_frames}")

        #Una vez que se han procesado todos los fotogramas, libera los recursos asociados con los videos y cierra todas las ventanas abiertas.
        video.release() 
        output_video.release()
        cv2.destroyAllWindows()
        #mprime un mensaje indicando que el proceso ha sido completado y la ubicación del video generado.
        print("Proceso completado. El video generado se encuentra en:", self.output_path)



#Verifica si el script está siendo ejecutado como un programa principal o si está siendo importado como un módulo en otro script.
if __name__ == "__main__":
    obj = CountObject2(videopath, output_path)
    obj.process_video()
