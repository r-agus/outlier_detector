import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import re
import time
from functools import lru_cache
import threading

# Intentar importar soporte para drag & drop
try:
    import tkinterdnd2
    HAS_DND = True
except ImportError:
    HAS_DND = False

# Clase base que se usará según disponibilidad de DnD
if HAS_DND:
    BaseClass = tkinterdnd2.TkinterDnD.Tk
else:
    print("No se ha podido importar tkinterdnd2. La funcionalidad de arrastrar y soltar no estará disponible.")
    BaseClass = tk.Tk

class SignalViewer(BaseClass):
    def __init__(self):
        super().__init__()
        
        self.title("Visualizador de Señales")
        self.geometry("1000x700")
        
        # Variables para almacenar datos
        self.signals = {}  # Diccionario para almacenar las señales {nombre_archivo: (tiempos, valores)}
        self.markers = []  # Lista para almacenar marcadores de tiempo
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Cache para datos procesados y estado de visualización
        self.processed_data_cache = {}
        self.current_view_center = None  # Para mantener el centro durante zoom
        self.is_panning = False  # Indicador para saber si estamos desplazándonos
        self.drag_in_progress = False  # Indicador específico para arrastres en curso
        self.pending_update = None  # Para gestionar actualizaciones pendientes
        
        # Referencias a ejes para optimizar el modo múltiple
        self.current_axes = {}  # Para mantener referencia a los ejes actuales
        self.last_mode = None   # Para detectar cambios de modo
        self.last_window_config = None  # Para detectar cambios en la estructura
        
        # Variables para blitting y optimización
        self.background = None  # Para almacenar el fondo de la figura
        self.artists = []  # Artistas gráficos que necesitan actualizarse
        
        # Variable para control de rendimiento
        self.max_points_per_plot = 5000  # Límite de puntos a dibujar por gráfica para mejor rendimiento
        self.preview_downsampling = 5  # Factor adicional de downsampling durante desplazamiento
        
        # Variable para el modo de visualización
        self.plot_mode = tk.StringVar(value="normal")
        # Variable para el modo de sincronización
        self.sync_mode = tk.StringVar(value="all")
        
        # Variable para controlar si se muestran las ventanas de 32ms
        self.draw_windows = tk.BooleanVar(value=False)
        
        self._create_widgets()
        
        # Configurar soporte de drag and drop si está disponible
        if HAS_DND:
            self.drop_target_register(tkinterdnd2.DND_FILES)
            self.dnd_bind('<<Drop>>', self.process_drop)
        
    def _create_widgets(self):
        # Frame principal dividido en dos
        self.main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo para controles
        self.control_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.control_frame, weight=1)
        
        # Panel derecho para la gráfica
        self.plot_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.plot_frame, weight=3)
        
        # === CONTROLES ===
        # Botón para cargar archivos
        self.load_button = ttk.Button(self.control_frame, text="Cargar Ficheros", command=self.load_files)
        self.load_button.pack(fill=tk.X, pady=(0, 5))
        
        # Botón para borrar todo
        self.clear_button = ttk.Button(self.control_frame, text="Borrar Todo", command=self.clear_all)
        self.clear_button.pack(fill=tk.X, pady=5)
        
        # Frame para opciones de visualización
        self.view_options_frame = ttk.LabelFrame(self.control_frame, text="Opciones de Visualización")
        self.view_options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Radiobutton(self.view_options_frame, text="Normal", variable=self.plot_mode, 
                        value="normal", command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.view_options_frame, text="Logarítmico", variable=self.plot_mode, 
                        value="log", command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.view_options_frame, text="Múltiples Gráficas", variable=self.plot_mode, 
                        value="multiple", command=self.update_plot).pack(anchor=tk.W, padx=5, pady=2)
        
        # Separador
        ttk.Separator(self.view_options_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Opción para dibujar ventanas de 32ms (movida al panel izquierdo)
        self.windows_check = ttk.Checkbutton(self.view_options_frame, text="Dibujar ventanas (32ms)",
                                           variable=self.draw_windows, command=self.update_plot)
        self.windows_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Separador adicional después de la opción de ventanas
        ttk.Separator(self.view_options_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Opciones de sincronización (solo activas en modo múltiple)
        ttk.Label(self.view_options_frame, text="Sincronización de ejes:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.view_options_frame, text="Todas las gráficas", variable=self.sync_mode, 
                        value="all", command=self.update_plot).pack(anchor=tk.W, padx=15, pady=2)
        ttk.Radiobutton(self.view_options_frame, text="Por columna", variable=self.sync_mode, 
                        value="column", command=self.update_plot).pack(anchor=tk.W, padx=15, pady=2)
        
        # Frame para marcador de tiempo
        self.marker_frame = ttk.LabelFrame(self.control_frame, text="Marcador de Tiempo")
        self.marker_frame.pack(fill=tk.X, pady=10)
        
        self.marker_value = tk.StringVar()
        self.marker_entry = ttk.Entry(self.marker_frame, textvariable=self.marker_value)
        self.marker_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.mark_button = ttk.Button(self.marker_frame, text="Marcar", command=self.add_marker)
        self.mark_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Lista de archivos cargados
        self.files_frame = ttk.LabelFrame(self.control_frame, text="Archivos Cargados")
        self.files_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.files_listbox = tk.Listbox(self.files_frame)
        self.files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Marco para estadísticas
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Estadísticas")
        self.stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = ttk.Label(self.stats_frame, text="Archivos: 0\nPuntos totales: 0")
        self.stats_label.pack(padx=5, pady=5)
        
        # === GRÁFICO ===
        # Crear figura de matplotlib
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.plot.set_xlabel('Tiempo (s)')
        self.plot.set_ylabel('Valor')
        self.plot.grid(True)
        
        # Integrar figura en Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mejorar el manejo de eventos para mayor fluidez
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('draw_event', self.on_draw)
        
        # Barra de herramientas de navegación
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Agregar un callback cuando cambia la vista
        # Esto es fundamental para capturar los cambios realizados con la barra de herramientas
        self.figure.canvas.mpl_connect('draw_event', self.on_draw)
        
        # Slider para desplazarse por la señal
        self.slider_frame = ttk.Frame(self.plot_frame)
        self.slider_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Botones de navegación para un uso más intuitivo
        self.nav_frame = ttk.Frame(self.slider_frame)
        self.nav_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(self.nav_frame, text="Desplazar:").pack(side=tk.LEFT, padx=5)
        
        # Crear contenedores para los botones de desplazamiento
        self.left_buttons_frame = ttk.Frame(self.nav_frame)
        self.left_buttons_frame.pack(side=tk.LEFT, padx=5)
        
        self.right_buttons_frame = ttk.Frame(self.nav_frame)
        self.right_buttons_frame.pack(side=tk.RIGHT, padx=5)
        
        # Tiempos de desplazamiento en segundos
        pan_times = [10, 5, 1, 0.5, 0.1]
        
        # Crear botones para desplazarse hacia la izquierda
        for pan_time in pan_times:
            btn = ttk.Button(self.left_buttons_frame, text=f"◀ {pan_time}s", width=6,
                           command=lambda t=pan_time: self.pan_left(t))
            btn.pack(side=tk.LEFT, padx=2)
        
        # Crear botones para desplazarse hacia la derecha
        for pan_time in pan_times:
            btn = ttk.Button(self.right_buttons_frame, text=f"{pan_time}s ▶", width=6,
                           command=lambda t=pan_time: self.pan_right(t))
            btn.pack(side=tk.RIGHT, padx=2)
        
        # Separación
        ttk.Separator(self.slider_frame, orient='horizontal').pack(fill='x', pady=3)
        
        ttk.Label(self.slider_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_scale = ttk.Scale(self.slider_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                  command=self.update_zoom)
        self.zoom_scale.set(100)
        self.zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def on_mouse_press(self, event):
        """Detecta el inicio de un desplazamiento o zoom"""
        if event.button == 1:  # Botón izquierdo
            self.drag_in_progress = True
            # Capturar el fondo para blitting
            self.background = self.canvas.copy_from_bbox(self.figure.bbox)
    
    def on_mouse_move(self, event):
        """Maneja el movimiento del mouse durante el desplazamiento"""
        # Solo procesar si hay un desplazamiento en curso y estamos en modo pan
        if self.drag_in_progress and self.toolbar.mode == 'pan/zoom':
            self.is_panning = True
            
            # Durante el desplazamiento, solo actualizar los límites de los ejes
            # sin redibujar completamente para mayor fluidez
            if hasattr(self, 'plot') and self.plot:
                x_min, x_max = self.plot.get_xlim()
                self.current_view_center = x_min + (x_max - x_min) / 2
    
    def on_mouse_release(self, event):
        """Detecta el fin de un desplazamiento y actualiza la vista con calidad completa"""
        if event.button == 1 and self.drag_in_progress:  # Botón izquierdo
            self.drag_in_progress = False
            
            # Si estábamos desplazándonos, actualizar con la nueva posición
            if self.is_panning:
                self.is_panning = False
                
                # Cancelar cualquier actualización pendiente
                if self.pending_update:
                    self.after_cancel(self.pending_update)
                
                # Programar la actualización completa con un pequeño retraso
                # para permitir que matplotlib complete sus operaciones internas
                self.pending_update = self.after(50, lambda: self.update_view_after_interaction())
    
    def on_draw(self, event):
        """Captura los eventos de redibujado de matplotlib"""
        # Capturar el fondo cuando se redibuja la figura
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
    
    def on_scroll(self, event):
        """Maneja el evento de la rueda del ratón para zoom"""
        # Cancelar cualquier actualización pendiente
        if self.pending_update:
            self.after_cancel(self.pending_update)
        
        # Programar actualización después de un pequeño retraso
        self.pending_update = self.after(100, lambda: self.update_view_after_interaction())
    
    def refresh_plot_without_rebuild(self, current_xlim):
        """
        Actualiza los datos en las gráficas existentes sin reconstruir toda la figura,
        lo que resulta en una actualización mucho más rápida.
        """
        if not self.signals or not current_xlim:
            return
            
        view_min, view_max = current_xlim
        mode = self.plot_mode.get()
        zoom_level = self.zoom_scale.get() / 100
        
        # Detectar cambio de modo (normal/log/multiple)
        mode_changed = (self.last_mode != mode)
        self.last_mode = mode
        
        try:
            # Si estamos en el modo de gráfica simple, la actualización es más sencilla
            if mode != "multiple":
                if hasattr(self, 'plot') and self.plot:
                    # Limpiar solo las líneas, no toda la figura
                    while self.plot.lines:
                        self.plot.lines[0].remove()
                    
                    # Redibujar las señales con los datos downsampled apropiados
                    for i, (filename, (times, values)) in enumerate(self.signals.items()):
                        color_idx = i % len(self.colors)
                        plot_times, plot_values = self.get_downsampled_data(
                            times, values, view_min, view_max, zoom_level)
                        self.plot.plot(plot_times, plot_values, color=self.colors[color_idx], linewidth=1,
                                   label=os.path.basename(filename))
                    
                    # Restaurar la leyenda
                    self.plot.legend(loc='upper right', fontsize=8)
                    
                    # Asegurar que los límites se mantengan
                    self.plot.set_xlim(view_min, view_max)
                    
                    # Dibujar ventanas de tiempo si está habilitado
                    self.draw_time_windows(self.plot, view_min, view_max)
                    
                    # Actualizar canvas de forma eficiente
                    self.canvas.draw_idle()
            else:
                # Para el modo múltiple, simplemente usar la reconstrucción completa
                # pero asegurando que los límites X se mantengan
                self.update_plot(force_reload=False, keep_position=True)
        except Exception as e:
            print(f"Error en actualización parcial: {e}")
            # En caso de error, volver al método completo
            self.update_plot(force_reload=True, keep_position=True)

    def update_view_after_interaction(self):
        """Actualiza la vista después de una interacción, evitando múltiples actualizaciones"""
        # Obtener la posición actual de la vista
        try:
            current_xlim = None
            
            # Intentar obtener los límites actuales de la vista
            if hasattr(self, 'plot') and self.plot:
                current_xlim = self.plot.get_xlim()
            elif len(self.figure.get_axes()) > 0:
                # En modo múltiple, usar el primer eje visible
                visible_axes = [ax for ax in self.figure.get_axes() if ax.get_visible()]
                if visible_axes:
                    current_xlim = visible_axes[0].get_xlim()
            
            if current_xlim:
                # Actualizar el centro de vista para coincidir con la vista actual
                view_min, view_max = current_xlim
                self.current_view_center = view_min + (view_max - view_min) / 2
                
                # Actualizar la vista con datos de alta calidad pero manteniendo la posición
                self.refresh_plot_without_rebuild(current_xlim)
            else:
                # Si no pudimos obtener los límites, hacer una actualización completa
                self.update_plot(force_reload=True, keep_position=False)
        except Exception as e:
            # En caso de error, volver al método de actualización completa
            print(f"Error en actualización optimizada: {e}")
            self.update_plot(force_reload=True, keep_position=True)
    
    def on_motion(self, event):
        """Detecta movimiento del mouse para identificar panning"""
        if self.toolbar.mode == 'pan/zoom' and event.button == 1:  # Si está en modo pan y con botón izquierdo
            self.is_panning = True
    
    def on_axes_leave(self, event):
        """Cuando el ratón sale de los ejes, verificar si estábamos haciendo pan"""
        if self.is_panning:
            # Programar una actualización después de un pequeño delay
            # para permitir que matplotlib complete su operación
            self.after(100, self.update_after_pan)
    
    def on_draw(self, event):
        """Detecta cuando matplotlib redibuja la figura (después de pan, zoom, etc.)"""
        if self.is_panning:
            self.is_panning = False
            # Programar actualización después de que matplotlib complete el redibujado
            self.after(100, self.update_after_pan)
    
    def update_after_pan(self):
        """Actualiza la vista después de un desplazamiento, preservando la posición actual"""
        if not self.signals:
            return
            
        # Obtener los límites actuales establecidos por la navegación
        if hasattr(self, 'plot') and self.plot:
            x_min, x_max = self.plot.get_xlim()
        else:
            # En modo múltiple, encontrar el primer eje visible
            for ax in self.figure.get_axes():
                if ax.get_visible():
                    x_min, x_max = ax.get_xlim()
                    break
            else:
                return  # No hay ejes visibles
        
        # Actualizar el centro de vista para que coincida con la vista actual
        self.current_view_center = x_min + (x_max - x_min) / 2
        
        # Actualizar con los datos extendidos, pero mantener la vista actual
        self.update_plot(force_reload=True, keep_position=True)
    
    def on_scroll(self, event):
        """Detecta cuando se usa la rueda del ratón para hacer zoom"""
        # Programar actualización después de un pequeño retraso
        self.after(150, self.update_after_pan)
    
    def load_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Seleccionar archivos de señal",
            filetypes=(("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*"))
        )
        
        if filepaths:
            for filepath in filepaths:
                self.load_file(filepath)
            
            self.update_plot()
    
    def process_drop(self, event):
        # Obtener rutas de archivos soltados
        filepaths = self.tk.splitlist(event.data)
        
        for filepath in filepaths:
            # En Windows, las rutas vienen con {} 
            filepath = filepath.strip('{}')
            self.load_file(filepath)
            
        self.update_plot()
    
    def load_file(self, filepath):
        try:
            # Obtener nombre de archivo
            filename = os.path.basename(filepath)
            
            # Leer datos
            times = []
            values = []
            
            with open(filepath, 'r') as file:
                for line in file:
                    # Usar expresión regular para manejar el formato específico
                    # "   4.1052200e+01  -7.5933762e+05"
                    match = re.match(r'\s*(\S+)\s+(\S+)', line)
                    if match:
                        time_str, value_str = match.groups()
                        try:
                            time = float(time_str)
                            value = float(value_str)
                            times.append(time)
                            values.append(value)
                        except ValueError:
                            continue
            
            if len(times) > 0:
                # Almacenar datos
                self.signals[filename] = (np.array(times), np.array(values))
                
                # Actualizar lista de archivos
                self.files_listbox.insert(tk.END, f"{filename} ({len(times)} puntos)")
                
                # Actualizar estadísticas
                self.update_stats()
                
                return True
            else:
                messagebox.showerror("Error", f"No se pudieron extraer datos válidos de {filename}")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo {filepath}: {str(e)}")
            return False
    
    @lru_cache(maxsize=32)
    def downsample_signal(self, times_tuple, values_tuple, view_min, view_max, extended_view_min, extended_view_max, points_per_second):
        """
        Reduce la cantidad de puntos en la señal para mejorar el rendimiento.
        Adaptativo según el nivel de zoom.
        Incluye margen extendido de 50% a cada lado.
        """
        times = np.array(times_tuple)
        values = np.array(values_tuple)
        
        # Primero filtrar solo los puntos en el rango extendido
        mask = (times >= extended_view_min) & (times <= extended_view_max)
        visible_times = times[mask]
        visible_values = values[mask]
        
        if len(visible_times) == 0:
            return np.array([]), np.array([])
        
        # Calcular cuántos puntos necesitamos para la densidad actual
        # Basado en la duración visible (sin la extensión)
        visible_duration = view_max - view_min
        try:
            # Usar puntos por segundo sobre la duración visible
            target_points = int(visible_duration * points_per_second)
            # Añadir puntos adicionales para las regiones extendidas
            extended_duration = extended_view_max - extended_view_min
            extension_factor = extended_duration / visible_duration
            target_points = int(target_points * extension_factor)
        except OverflowError:
            target_points = self.max_points_per_plot

        # Si hay menos puntos de los necesarios, devolver todos
        if len(visible_times) <= target_points:
            return visible_times, visible_values
        
        # Calcular factor de downsampling
        factor = int(len(visible_times) / target_points)
        
        # Asegurar que al menos tomamos algunos puntos
        factor = max(1, factor)
        
        # Devolver datos reducidos utilizando slicing con stride
        return visible_times[::factor], visible_values[::factor]
    
    def get_downsampled_data(self, times, values, view_min, view_max, zoom_level):
        """
        Wrapper para la función cacheada de downsampling con zoom adaptativo.
        Incluye 50% extra a cada lado del rango visible.
        """
        # Aplicar un mayor downsampling durante el desplazamiento para mayor fluidez
        actual_points_factor = 1
        if self.is_panning or self.drag_in_progress:
            actual_points_factor = self.preview_downsampling
        
        # Calcular puntos por segundo basados en el nivel de zoom
        if zoom_level <= 0.05:  # Si estamos cerca del zoom máximo
            # Mostrar todos los puntos en el rango visible
            points_per_second = float('inf')
        else:
            # Interpolar entre 10 y 1000 puntos por segundo según el zoom
            points_per_second = (1000 - 990 * zoom_level) / actual_points_factor
        
        # Calcular rango extendido (50% más a cada lado)
        visible_range = view_max - view_min
        extension = visible_range * 0.5
        extended_view_min = max(np.min(times), view_min - extension)
        extended_view_max = min(np.max(times), view_max + extension)
        
        # Convertir arrays a tuplas para poder usarlos como claves en caché
        times_tuple = tuple(times.tolist())
        values_tuple = tuple(values.tolist())
        
        # Usar la función cacheada con el rango extendido
        return self.downsample_signal(times_tuple, values_tuple, view_min, view_max, 
                                     extended_view_min, extended_view_max, points_per_second)
    
    def update_plot(self, force_reload=False, keep_position=False):
        """
        Actualiza el gráfico.
        - force_reload: Fuerza recarga con márgenes extendidos
        - keep_position: Mantiene la posición actual de la vista
        """
        start_time = time.time()  # Para medir rendimiento
        
        # Guardar la posición actual si es necesario
        current_xlim = None
        if keep_position and len(self.figure.get_axes()) > 0:
            # Obtener límites del primer eje visible
            for ax in self.figure.get_axes():
                if ax.get_visible():
                    current_xlim = ax.get_xlim()
                    break
        
        # Limpiar gráfico
        self.figure.clear()
        
        if not self.signals:
            self.canvas.draw()
            return
        
        mode = self.plot_mode.get()
        
        # Actualizar el estado actual para optimizaciones
        self.last_mode = mode
        
        # Determinar rango global de tiempo
        min_time = float('inf')
        max_time = float('-inf')
        
        for _, (times, _) in self.signals.items():
            if len(times) > 0:
                min_time = min(min_time, np.min(times))
                max_time = max(max_time, np.max(times))
        
        # Calcular zoom y centro de vista
        zoom_level = self.zoom_scale.get() / 100
        visible_range = (max_time - min_time) * zoom_level
        
        # Si hay una posición guardada y estamos en keep_position, usarla
        if keep_position and current_xlim:
            view_min, view_max = current_xlim
            # Solo actualizar el centro si no es la primera vez
            if self.current_view_center is not None:
                self.current_view_center = view_min + (view_max - view_min) / 2
        else:
            # Calcular basado en el centro actual
            if self.current_view_center is None:
                # Primera vez - centro en el centro del rango completo
                self.current_view_center = min_time + (max_time - min_time) / 2
            
            # Calcular límites basados en el centro actual
            view_min = max(min_time, self.current_view_center - visible_range / 2)
            view_max = min(max_time, view_min + visible_range)
            
            # Ajustar centro si llegamos a los límites
            if view_min == min_time:
                view_max = view_min + visible_range
            elif view_max == max_time:
                view_min = view_max - visible_range
            
        # Actualizar el centro de la vista para la próxima actualización
        self.current_view_center = view_min + (view_max - view_min) / 2
        
        if mode == "multiple":
            # Organizar señales por descarga y número
            organized_signals = self.organize_signals()
            
            if not organized_signals:
                self.canvas.draw()
                return
            
            # Crear matriz de subplots
            n_discharges = len(organized_signals)
            max_signals = max(len(signals) for signals in organized_signals.values())
            
            # Ordenar descargas alfabéticamente para consistencia
            discharge_names = sorted(organized_signals.keys())
            
            # Configurar cómo se comparten los ejes X
            if self.sync_mode.get() == "all":
                axes = self.figure.subplots(max_signals, n_discharges, sharex=True)
            else:  # "column"
                axes = self.figure.subplots(max_signals, n_discharges, sharex='col')
            
            # Asegurar que axes sea siempre un array 2D incluso con una sola fila o columna
            if n_discharges == 1 and max_signals == 1:
                axes = np.array([[axes]])
            elif n_discharges == 1:
                axes = axes.reshape(-1, 1)
            elif max_signals == 1:
                axes = axes.reshape(1, -1)
            
            # Recorrer cada descarga (columna)
            for col, discharge_name in enumerate(discharge_names):
                signals = organized_signals[discharge_name]
                
                # Añadir título a la columna (optimizado para no sobrecargar la figura)
                if col < len(discharge_names):
                    self.figure.text(
                        (col + 0.5) / max(1, n_discharges), 0.98, 
                        f"descarga: {discharge_name}", 
                        ha='center', va='top', fontsize=9, fontweight='bold'
                    )
                
                # Recorrer cada señal en la descarga
                for row, (signal_num, filename, times, values) in enumerate(signals):
                    if row >= max_signals or col >= n_discharges:
                        continue  # Protección contra errores de índice
                        
                    ax = axes[row, col]
                    color_idx = (row * n_discharges + col) % len(self.colors)
                    
                    # Aplicar downsampling adaptativo según el zoom
                    plot_times, plot_values = self.get_downsampled_data(
                        times, values, view_min, view_max, zoom_level)
                    
                    ax.plot(plot_times, plot_values, color=self.colors[color_idx], linewidth=1)
                    
                    # Establecer límites centrados usando la vista calculada
                    ax.set_xlim(view_min, view_max)
                    
                    # Dibujar ventanas de tiempo si está habilitado
                    self.draw_time_windows(ax, view_min, view_max)
                    
                    # Añadir marcadores de tiempo (solo si están dentro del rango visible)
                    visible_markers = [m for m in self.markers if view_min <= m <= view_max]
                    for marker_time in visible_markers:
                        ax.axvline(x=marker_time, color='red', linestyle='--', linewidth=1)
                        y_pos = ax.get_ylim()[1] * 0.95  # Posición más optimizada
                        ax.text(marker_time, y_pos, f"{marker_time:.2f}s", 
                                rotation=90, va='top', ha='right', color='red',
                                fontsize=8)  # Texto más pequeño para mejor rendimiento
                    
                    # Configurar gráfico - reducir detalles para mejor rendimiento
                    ax.set_title(f"Señal {signal_num}", fontsize=8)  # Título simplificado
                    ax.grid(True, alpha=0.5)  # Grid menos prominente
                    
                    # Etiquetas solo en los bordes
                    if row == len(signals) - 1:
                        ax.set_xlabel('Tiempo (s)')
                    if col == 0:
                        ax.set_ylabel('Valor')
                
                # Ocultar subplots vacíos
                for row in range(len(signals), max_signals):
                    if row < max_signals and col < n_discharges:
                        axes[row, col].set_visible(False)
            
            # Optimización: tight_layout con menos restricciones
            self.figure.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5, h_pad=0.5, w_pad=0.5)
            
            # Guardar la estructura actual para optimizaciones
            discharge_names = sorted(organized_signals.keys())
            self.last_window_config = (
                tuple(discharge_names), 
                tuple(len(organized_signals[w]) for w in discharge_names),
                self.sync_mode.get()
            )
            
        else:
            # Modo normal o logarítmico con optimización de rendimiento
            self.plot = self.figure.add_subplot(111)
            
            for i, (filename, (times, values)) in enumerate(self.signals.items()):
                color_idx = i % len(self.colors)
                
                # Aplicar downsampling adaptativo según el zoom
                plot_times, plot_values = self.get_downsampled_data(
                    times, values, view_min, view_max, zoom_level)
                
                self.plot.plot(plot_times, plot_values, color=self.colors[color_idx], linewidth=1, 
                           label=os.path.basename(filename))
            
            # Aplicar escala logarítmica si está seleccionado ese modo
            if mode == "log":
                self.plot.set_yscale('log')
            
            # Establecer límites centrados
            self.plot.set_xlim(view_min, view_max)
            
            # Dibujar ventanas de tiempo si está habilitado
            self.draw_time_windows(self.plot, view_min, view_max)
            
            # Añadir marcadores de tiempo (solo si están dentro del rango visible)
            visible_markers = [m for m in self.markers if view_min <= m <= view_max]
            for marker_time in visible_markers:
                self.plot.axvline(x=marker_time, color='red', linestyle='--', linewidth=1)
                y_pos = self.plot.get_ylim()[1] * 0.95
                self.plot.text(marker_time, y_pos, f"{marker_time:.2f}s", 
                          rotation=90, va='top', ha='right', color='red', fontsize=8)
            
            # Configurar gráfico
            self.plot.grid(True, alpha=0.5)
            self.plot.set_xlabel('Tiempo (s)')
            self.plot.set_ylabel('Valor')
            self.plot.legend(loc='upper right', fontsize=8)
        
        # Si estamos manteniendo la posición, restaurarla después de redibujar
        if keep_position and current_xlim:
            view_min, view_max = current_xlim
            # Actualizar el centro actual sin cambiar la vista
            self.current_view_center = view_min + (view_max - view_min) / 2
        
        # Al final, si estamos manteniendo la posición, forzar los límites guardados
        if keep_position and current_xlim and mode == "multiple":
            for ax in self.figure.get_axes():
                if ax.get_visible():
                    ax.set_xlim(current_xlim)
        elif keep_position and current_xlim:
            self.plot.set_xlim(current_xlim)
        
        # Al finalizar, borrar cualquier actualización pendiente
        self.pending_update = None
        
        # Registrar el tiempo que tomó la actualización
        elapsed = time.time() - start_time
        if elapsed > 0.5:  # Si tarda más de medio segundo, es lento
            print(f"Actualización lenta: {elapsed:.3f} segundos")
        
        # Optimizar redibujado
        self.canvas.draw_idle()
    
    def update_zoom(self, event=None):
        """Actualiza el zoom y fuerza recarga con margen extendido"""
        self.update_plot(force_reload=True, keep_position=False)  # No mantener posición en cambio de zoom
    
    def add_marker(self):
        try:
            time_value = float(self.marker_value.get().replace(',', '.'))
            self.markers.append(time_value)
            self.update_plot()
            self.marker_value.set("")  # Limpiar campo
        except ValueError:
            messagebox.showerror("Error", "Por favor, introduzca un valor de tiempo válido.")
    
    def clear_all(self):
        self.signals.clear()
        self.markers.clear()
        self.files_listbox.delete(0, tk.END)
        self.current_view_center = None  # Reiniciar centro de vista
        self.processed_data_cache = {}  # Limpiar caché
        self.update_stats()
        self.update_plot()
    
    def update_stats(self):
        num_files = len(self.signals)
        total_points = sum(len(times) for times, _ in self.signals.values())
        self.stats_label.config(text=f"Archivos: {num_files}\nPuntos totales: {total_points}")
    
    def parse_filename(self, filename):
        """Extrae la descarga (YYYY) y el número de señal (ZZ) del nombre del archivo"""
        # Formato: XXX_YYYY_ZZ_WW_ttttttt.txt
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                discharge_name = parts[1]
                signal_num = int(parts[2])
                return discharge_name, signal_num
            except (IndexError, ValueError):
                pass
        return None, None
    
    def organize_signals(self):
        """Organiza las señales por descarga y número de señal"""
        organized = {}
        
        for filename, (times, values) in self.signals.items():
            discharge_name, signal_num = self.parse_filename(filename)
            if discharge_name is not None:
                if discharge_name not in organized:
                    organized[discharge_name] = []
                organized[discharge_name].append((signal_num, filename, times, values))
            else:
                # Si no se puede parsear el nombre, usar un grupo genérico
                if "unknown" not in organized:
                    organized["unknown"] = []
                # Usar índice como número de señal para archivos que no siguen el formato
                signal_num = len(organized["unknown"])
                organized["unknown"].append((signal_num, filename, times, values))
        
        # Ordenar cada descarga por número de señal
        for discharge_name in organized:
            organized[discharge_name].sort()  # Ordenar por signal_num (primer elemento de la tupla)
        
        return organized
    
    def pan_left(self, pan_amount=5):
        """Desplaza la vista a la izquierda"""
        if self.current_view_center is not None:
            zoom_level = self.zoom_scale.get() / 100
            min_time = float('inf')
            max_time = float('-inf')
            for _, (times, _) in self.signals.items():
                if len(times) > 0:
                    min_time = min(min_time, np.min(times))
                    max_time = max(max_time, np.max(times))
            
            visible_range = (max_time - min_time) * zoom_level
            self.current_view_center = max(min_time + visible_range/2, self.current_view_center - pan_amount)
            self.update_plot(force_reload=True, keep_position=False)  # No mantener posición en desplazamiento manual
    
    def pan_right(self, pan_amount=5):
        """Desplaza la vista a la derecha"""
        if self.current_view_center is not None:
            zoom_level = self.zoom_scale.get() / 100
            min_time = float('inf')
            max_time = float('-inf')
            for _, (times, _) in self.signals.items():
                if len(times) > 0:
                    min_time = min(min_time, np.min(times))
                    max_time = max(max_time, np.max(times))
            
            visible_range = (max_time - min_time) * zoom_level
            self.current_view_center = min(max_time - visible_range/2, self.current_view_center + pan_amount)
            self.update_plot(force_reload=True, keep_position=False)  # No mantener posición en desplazamiento manual
    
    def draw_time_windows(self, ax, view_min, view_max):
        """Dibuja ventanas de tiempo de 32ms alternando colores"""
        if not self.draw_windows.get():
            return
            
        # Tamaño de ventana en segundos
        window_size = 0.032  # 32 ms
        
        # Determinar el tiempo mínimo absoluto de todas las señales
        min_time = float('inf')
        for _, (times, _) in self.signals.items():
            if len(times) > 0:
                min_time = min(min_time, np.min(times))
        
        # Calcular cuántas ventanas completas hay desde el tiempo mínimo hasta el rango visible
        # y determinar el inicio de la ventana que contiene view_min
        windows_from_min = int((view_min - min_time) / window_size)
        start_window = min_time + (windows_from_min * window_size)
        
        # Ajustar para asegurar que empezamos con una ventana que sea visible
        if start_window > view_min:
            start_window = start_window - window_size
            
        # Determinar si la ventana inicial es par o impar basado en su posición desde el tiempo mínimo
        is_odd = (windows_from_min % 2 == 1)
        
        # Dibujar ventanas alternando colores
        window_start = start_window
        colors = ['#1f77b440', '#ff7f0e40']
        
        while window_start < view_max:
            window_end = min(window_start + window_size, view_max)

            color_idx = int(is_odd) % 2
            
            # Rectángulo sombreado
            rect = ax.axvspan(window_start, window_end, 
                             color=colors[color_idx], 
                             alpha=0.2,  # Transparencia adicional
                             zorder=0)   # Asegurar que esté detrás de las señales
            
            # Siguiente ventana
            window_start = window_end
            is_odd = not is_odd

if __name__ == "__main__":
    app = SignalViewer()
    app.mainloop()