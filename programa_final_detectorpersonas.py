import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import subprocess
import threading
from PIL import Image, ImageTk
from detector_personas_id4 import CountObject2

class DetectorPersonasApp:
    #Defino una funcion que me ayudara a generar la ventana de la interfaz grafica con las demas añadiduras como texto, botones, cuadros de video hasta imagenes.
    def __init__(self, root):
        #líneas de código inicializan y configuran la ventana principal de la interfaz gráfica del programa "Detector de Personas" con un título específico, tamaño predefinido, color de fondo y configuración de fuente predeterminada.
        self.root = root
        self.root.title("PROGRAMA DETECTOR DE PERSONAS")
        self.root.geometry("800x600")
        self.root.config(bg="#ADD8E6")
        self.root.option_add('*Font', 'times 13')

        # Encabezado en la interfaz grafica con texto de datos informativos.
        self.encabezado_label = tk.Label(root, text="DETECTOR DE PERSONAS", font=("Times New Roman", 13, "bold"), bg="yellow")
        self.encabezado_label.pack()
        self.proyecto_label = tk.Label(root, text="PROYECTO CURSO DE DEEP LEARNING", font=("Times New Roman", 13), bg="white")
        self.proyecto_label.pack()
        self.universidad_label = tk.Label(root, text="UNIVERSIDAD TECNICA DE AMBATO / IEEE", font=("Times New Roman", 13), bg="white")
        self.universidad_label.pack()
        self.autor_label = tk.Label(root, text="JEAN CARLOS LEÓN", font=("Times New Roman", 13), bg="white")
        self.autor_label.pack()


        # Imagen en la parte superior izquierda de la interfaz gráfica
        self.img_izquierda = Image.open("imagen1.png") #cargo la imagen deseada
        self.img_izquierda = self.img_izquierda.resize((150, 100), Image.ANTIALIAS)
        self.img_izquierda = ImageTk.PhotoImage(self.img_izquierda) #ubicacion de la imagen
        self.imagen_izquierda = tk.Label(root, image=self.img_izquierda, bg="#ADD8E6")  # Fondo celeste
        self.imagen_izquierda.place(x=10, y=40)  # Ubicación en la parte superior izquierda

        # Imagen en la esquina superior derecha
        self.img_derecha = Image.open("imagen2.png") #cargo la imagen deseada
        self.img_derecha = self.img_derecha.resize((150, 100), Image.ANTIALIAS) #ubicacion de la imagen
        self.img_derecha = ImageTk.PhotoImage(self.img_derecha)
        self.imagen_derecha = tk.Label(root, image=self.img_derecha, bg="#ADD8E6")  # Fondo celeste
        self.imagen_derecha.place(x=650, y=40)  # Ubicación en la parte superior derecha

        # Etiqueta (texto) y botones para el procesamiento del video en la interfaz de usuario

        self.video_frame = tk.Frame(root, bg="white", bd=2, relief=tk.GROOVE, width=700) #creación del cuadro para reproducir el video
        self.video_frame.pack(pady=9)

        #Cuadro de texto y titulo del cuadro para asigar los botones.
        self.video_label = tk.Label(self.video_frame, text="OPCIONES DE CONFIGURACIÓN DE VIDEO: ", font=("Times New Roman", 15, "bold"), bg="white")
        self.video_label.pack(pady=10)

        #Botones del cuadro de texto
        self.btn_cargar_video = tk.Button(self.video_frame, text="Cargar Video", command=self.cargar_video, bg="YELLOW", fg="black")
        self.btn_cargar_video.pack()

        self.btn_procesar_video = tk.Button(self.video_frame, text="Procesar Video", command=self.procesar_video, bg="YELLOW", fg="black")
        self.btn_procesar_video.pack()

        self.btn_visualizar_video = tk.Button(self.video_frame, text="Visualizar Video Procesado / REANUDAR VIDEO ", command=self.visualizar_video, bg="blue", fg="white")
        self.btn_visualizar_video.pack()
        
        #boton salir para cerrar el programa
        self.btn_salir = tk.Button(root, text="Salir", command=root.quit, bg="red", fg="black")
        self.btn_salir.pack(side=tk.BOTTOM)

        # Botón para detener o continuar la reproducción del video
        self.btn_detener_continuar = tk.Button(root, text="DETENER VIDEO", command=self.detener_continuar_video, bg="red", fg="white")
        self.btn_detener_continuar.pack(pady=20)

        # Creamos un lienzo para mostrar el video
        self.canvas = tk.Canvas(root, width=800, height=480)
        self.canvas.pack()

        # Nombre del archivo de video que se genera del procesamiento por el algoritmo
        self.video_file = "resultado2.mp4"
       
        self.paused = False  # Variable para controlar si la reproducción del video está pausada o no

    #defino una función para poder cargar el video
    def cargar_video(self):
        self.filename = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4"), ("All files", "*.*"))) #se abrira una ventana de dialogo para poder cargar el video en la ubicación donde se encuentre el video
        #sentencia para capturar un frame del video
        if self.filename:
            self.cap = cv2.VideoCapture(self.filename)

        # Capturar el primer fotograma del video
        ret, frame = self.cap.read()

        # Obtener las dimensiones del lienzo para ajustarlo a la interfaz grafica
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Redimensionar el fotograma para que se ajuste al lienzo
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (canvas_width, canvas_height))

        # Mostrar el fotograma en una ventana separada
        cv2.imshow('Captura de video cargado', frame)


    #defino una función para poder procesar el video por el algoritmo
    def procesar_video(self):
    # Crea una nueva ventana emergente para mostrar la alerta
        self.alerta = tk.Toplevel(self.root)
        self.alerta.title("ALERTA")
        self.alerta.geometry("300x300")
        self.alerta_label = tk.Label(self.alerta, text="¡¡¡EL PROCESAMIENTO DEL VIDEO ESTA EN CURSO!!!")
        self.alerta_label.pack(pady=20)

    # Función para ejecutar el script en un hilo separado
        def ejecutar_script():

            programa = "detector_personas_id4.py"
            # Ejecuta el programa en un proceso secundario
            proceso = subprocess.Popen(["python", programa])
            # Espera a que el proceso secundario finalice
            proceso.wait()
            # Cierra la ventana emergente después de que el proceso haya terminado
            self.alerta.destroy()

        # Ejecuta la función en un hilo separado
        threading.Thread(target=ejecutar_script).start()
    #defino la función para visualizar el video resultante del procesamiento para poder cargar en el lienso o cuadro para visualizar el video
    def visualizar_video(self):
        try:
            # Abrir el archivo de video
            cap = cv2.VideoCapture(self.video_file)

            # Obtener las dimensiones del lienzo
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Bucle para mostrar los fotogramas del video
            while True:
                ret, frame = cap.read()
                if not ret:  # Si no hay más fotogramas en el video
                    break

                # Redimensionar el fotograma para que se ajuste al lienzo o en la caja donde se mostrara el video.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (canvas_width, canvas_height))

                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                # Mostrar el fotograma en el lienzo
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.canvas.image = frame
                self.root.update()  # Actualizar la interfaz

                # Permitir la interacción con el botón "Detener / Continuar Video"
                self.root.update_idletasks()

                # Salir del bucle si la reproducción del video está pausada
                if self.paused:
                    break

            # Liberar los recursos
            cap.release()
        #mensaje de error en caso de no ejecutarse correctamente el video.
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
   
   
    #funcion para poder detener y controlar el video que se reproducira
    def detener_continuar_video(self):
        # Cambiar el estado de la variable paused para detener o continuar la reproducción del video
        self.paused = not self.paused
        
#Verifica si el script está siendo ejecutado como un programa principal o si está siendo importado como un módulo en otro script.
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorPersonasApp(root)
    root.mainloop()
