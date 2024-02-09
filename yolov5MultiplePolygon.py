import numpy as np
import supervision as sv
import torch
import argparse

#Añado las variables que me generan el video para la entrada y la salida 
input_video = 'demo2.mp4'
output_video = 'resultado1.mp4'  # Este sera el video resultante despues del procesamiento del algoritmo


class CountObject():
    #Defino una funcion init para inicializar los atributos de la instancia. En este caso, 
    #el método __init__ toma dos argumentos: input_video_path y output_video_path, que representan las rutas de entrada 
    #y salida del video, respectivamente.
    def __init__(self,input_video_path,output_video_path) -> None:
        
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6') #carga un modelo YOLOv5 preentrenado utilizando la biblioteca Ultralytics
        self.colors = sv.ColorPalette.default() #crea una paleta de colores que se utilizará para resaltar los objetos detectados en el vídeo

        #asigno las rutas de entrada y de salida de la instancia actual
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        #inicializo las variable que contendra las zonas donde quiero el conteo de personas, los poligonos
        #ademas agrego un tipo de arreglo con numpy con los vertices de los poligonos escogidos del video.
        self.polygons = [
            np.array([
                [540,  985 ],
                [1620, 985 ],
                [2160, 1920],
                [1620, 2855],
                [540,  2855],
                [0,    1920]
            ], np.int32),
            np.array([
                [0,    1920],
                [540,  985 ],
                [0,    0   ]
            ], np.int32),
            np.array([
                [1620, 985 ],
                [2160, 1920],
                [2160,    0]
            ], np.int32),
            np.array([
                [540,  985 ],
                [0,    0   ],
                [2160, 0   ],
                [1620, 985 ]
            ], np.int32),
            np.array([
                [0,    1920],
                [0,    3840],
                [540,  2855]
            ], np.int32),
            np.array([
                [2160, 1920],
                [1620, 2855],
                [2160, 3840]
            ], np.int32),
            np.array([
                [1620, 2855],
                [540,  2855],
                [0,    3840],
                [2160, 3840]
            ], np.int32)
        ]

        #crear zonas de interés delimitadas por polígonos en un vídeo, utilizando la información del vídeo de cada frame 
        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        #inicializa la variable self.zones como una lista de zonas de interés. Cada zona de interés está definida por un polígono que delimita el área de la zona en el vídeo.
        self.zones = [
            sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=self.video_info.resolution_wh #La resolución del fotograma del vídeo, que se obtiene del objeto self.video_info
            )
            for polygon
            in self.polygons
        ]
        #se utilizan para crear anotadores de zonas de interés que serán utilizados para visualizar y anotar cada una de las zonas definidas en el vídeo. Cada anotador de 
        #zona tiene su propio color, grosor de línea y grosor de texto, lo que permite distinguir y resaltar las diferentes zonas de interés en el vídeo.
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone, 
                color=self.colors.by_idx(index), #color que se utilizará para visualizar la caja 
                thickness=6, #El grosor de la línea que se dibujará para representar la caja.
                text_thickness=8, # l grosor del texto que se superpondrá en la caja.
                text_scale=4  #determina la escala del texto
            )
            for index, zone
            in enumerate(self.zones)
        ]
        #se utilizan para crear anotadores de cajas que sera la cual ayuda al conteo de las personas que serán utilizados para visualizar y anotar cada una de las regiones delimitadas por polígonos en el vídeo. Cada anotador 
        #de caja tiene su propio color, grosor de línea y grosor de texto, lo que permite distinguir y resaltar las diferentes regiones de interés en el vídeo.
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index), 
                thickness=4, 
                text_thickness=4, 
                text_scale=2
                )
            for index
            in range(len(self.polygons))
        ]


    #De la funcion process_frame lo utilizo para  procesa un fotograma del vídeo, realiza la detección de objetos, filtra las detecciones para mantener solo las de interés, y luego agrega anotaciones visuales de las zonas de interés 
    #y las cajas delimitadoras de las detecciones al fotograma antes de devolverlo.
        
    def process_frame(self,frame: np.ndarray, i) -> np.ndarray:
        # detección de objetos en el fotograma actual
        results = self.model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)] #filtra las detecciones para mantener solo las que corresponden a la clase de objeto específico tienen una confianza (probabilidad de detección) superior a 0.5.

        #Genero un ciclo for para poder iterar cada frame y se mantenga la zona de interes
        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True) #Representa el anotador de la caja correspondiente a esa zona
            frame = zone_annotator.annotate(scene=frame) #Representa el anotador de zona correspondiente a esa zona.

        return frame
    
    #inicia el proceso de procesamiento del vídeo utilizando la función process_video de la biblioteca scikit-video, proporcionando la ruta de entrada y salida del vídeo, así como una función de devolución de llamada para procesar cada fotograma.
    def process_video(self):

        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame) #función de la biblioteca scikit-video que se utiliza para procesar un vídeo

#Es una construcción que determina si el script actual se está ejecutando como un programa principal o si se está importando como un módulo en otro script. Cuando se ejecuta como un programa principal, el bloque de código indentado debajo de esta línea se ejecuta automáticamente
#Esto es útil para definir el comportamiento específico que deseamos cuando el script se llama directamente desde la línea de comandos        
if __name__ == "__main__":

     obj = CountObject(input_video,output_video)
     obj.process_video()

