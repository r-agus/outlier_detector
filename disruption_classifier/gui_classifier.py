import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import os
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import the classifier from the existing file
from classifier import DisruptionClassifier, DISCHARGE_DATA_PATH, DISCHARGES_ETIQUETATION_DATA_FILE

class DisruptionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Disruption Classifier")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize the classifier
        self.classifier = DisruptionClassifier(DISCHARGE_DATA_PATH, DISCHARGES_ETIQUETATION_DATA_FILE)
        self.results = None
        
        # Track discharge IDs for training and test sets
        self.training_ids = set()
        self.test_ids = set()
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready to start")
        self.progress_var = tk.DoubleVar(value=0)
        
        # Initialize default model parameters - MOVED THIS EARLIER
        self.init_model_parameters()
        
        # Create main frames
        self.create_main_frames()
        
        # Create widgets
        self.create_control_panel()
        self.create_status_frame()
        self.create_results_area()
        
        # Track the current step
        self.current_step = 0
        self.update_step_indicators()
    
    def init_model_parameters(self):
        """Initialize model parameters with default values"""
        # OCSVM parameters
        self.ocsvm_params = {
            'kernel': tk.StringVar(value='rbf'),
            'nu': tk.DoubleVar(value=0.1),
            'gamma': tk.StringVar(value='auto')
        }
        
        # Isolation Forest parameters
        self.iforest_params = {
            'n_estimators': tk.IntVar(value=100),
            'contamination': tk.DoubleVar(value=0.1),
            'max_features': tk.DoubleVar(value=1.0)
        }
        
        # LSTM parameters
        self.lstm_params = {
            'layer1_units': tk.IntVar(value=64),
            'layer2_units': tk.IntVar(value=32),
            'dropout_rate': tk.DoubleVar(value=0.2),
            'learning_rate': tk.DoubleVar(value=0.001),
            'batch_size': tk.IntVar(value=16),
            'epochs': tk.IntVar(value=100)
        }
    
    def create_main_frames(self):
        # Main layout with three sections
        self.top_frame = ttk.Frame(self.root, padding="10")
        self.top_frame.pack(fill=tk.X)
        
        self.middle_frame = ttk.Frame(self.root)
        self.middle_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        self.control_frame = ttk.LabelFrame(self.middle_frame, text="Control Panel", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for results
        self.results_frame = ttk.LabelFrame(self.middle_frame, text="Results", padding="10")
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom panel for status
        self.status_frame = ttk.Frame(self.root, padding="10")
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    def create_control_panel(self):
        # Step indicators
        self.steps_frame = ttk.LabelFrame(self.control_frame, text="Workflow Steps", padding="10")
        self.steps_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.step_labels = []
        self.step_indicators = []
        
        steps = [
            "1. Load Data", 
            "2. Train Models",
            "3. Analyze Results",
            "4. Make Predictions"
        ]
        
        for i, step in enumerate(steps):
            frame = ttk.Frame(self.steps_frame)
            frame.pack(fill=tk.X, pady=2)
            
            indicator = ttk.Label(frame, text="○", font=("Arial", 12), width=2)
            indicator.pack(side=tk.LEFT)
            
            label = ttk.Label(frame, text=step)
            label.pack(side=tk.LEFT, padx=5)
            
            self.step_indicators.append(indicator)
            self.step_labels.append(label)
        
        # Control buttons
        self.buttons_frame = ttk.Frame(self.control_frame)
        self.buttons_frame.pack(fill=tk.X, pady=10)
        
        # Create tabs for basic controls and advanced parameters
        self.control_tabs = ttk.Notebook(self.control_frame)
        self.control_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Basic controls tab
        self.basic_tab = ttk.Frame(self.control_tabs)
        self.control_tabs.add(self.basic_tab, text="Basic Controls")
        
        # Parameters tab
        self.params_tab = ttk.Frame(self.control_tabs)
        self.control_tabs.add(self.params_tab, text="Model Parameters")
        
        # Add basic controls to the basic tab
        self.create_basic_controls(self.basic_tab)
        
        # Add parameter controls to the parameters tab
        self.create_parameter_controls(self.params_tab)
    
    def create_basic_controls(self, parent):
        # Data loading
        self.load_data_btn = ttk.Button(
            parent, 
            text="Load Discharge Data", 
            command=self.load_data
        )
        self.load_data_btn.pack(fill=tk.X, pady=5)
        
        # Training
        self.train_models_btn = ttk.Button(
            parent, 
            text="Train Models", 
            command=self.train_models,
            state=tk.DISABLED
        )
        self.train_models_btn.pack(fill=tk.X, pady=5)
        
        # Testing slider
        self.test_size_frame = ttk.Frame(parent)
        self.test_size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.test_size_frame, text="Test Size:").pack(side=tk.LEFT)
        
        self.test_size_var = tk.DoubleVar(value=0.3)
        self.test_size_slider = ttk.Scale(
            self.test_size_frame, 
            from_=0.1, 
            to=0.5, 
            orient=tk.HORIZONTAL,
            variable=self.test_size_var
        )
        self.test_size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(self.test_size_frame, text="0.3").pack(side=tk.LEFT)
        
        # Analysis
        self.analyze_btn = ttk.Button(
            parent, 
            text="Analyze Test Results", 
            command=self.analyze_results,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(fill=tk.X, pady=5)
        
        # Prediction
        self.predict_frame = ttk.Frame(parent)
        self.predict_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.predict_frame, text="Discharge ID:").pack(side=tk.LEFT)
        
        # Replace text entry with combobox for discharge selection
        self.discharge_id_var = tk.StringVar()
        self.discharge_combobox = ttk.Combobox(
            self.predict_frame, 
            textvariable=self.discharge_id_var,
            state="readonly"
        )
        self.discharge_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add a binding for filtering as user types
        self.discharge_combobox.bind('<KeyRelease>', self.filter_discharge_list)
        
        self.predict_btn = ttk.Button(
            self.predict_frame, 
            text="Predict", 
            command=self.predict_discharge,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.RIGHT)
    
    def create_parameter_controls(self, parent):
        # Verify parameters were properly initialized
        if not hasattr(self, 'ocsvm_params') or not hasattr(self, 'iforest_params') or not hasattr(self, 'lstm_params'):
            self.init_model_parameters()  # Reinitialize if missing
        
        # Create notebook for model parameters
        param_notebook = ttk.Notebook(parent)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # OCSVM parameters tab
        ocsvm_tab = ttk.Frame(param_notebook)
        param_notebook.add(ocsvm_tab, text="One-Class SVM")
        
        # OCSVM parameters
        ttk.Label(ocsvm_tab, text="Kernel:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_options = ['rbf', 'linear', 'poly', 'sigmoid']
        kernel_dropdown = ttk.Combobox(ocsvm_tab, textvariable=self.ocsvm_params['kernel'], values=kernel_options, state="readonly")
        kernel_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(ocsvm_tab, text="Nu (0.0-1.0):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        nu_slider = ttk.Scale(ocsvm_tab, from_=0.01, to=1.0, variable=self.ocsvm_params['nu'], orient=tk.HORIZONTAL)
        nu_slider.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(ocsvm_tab, textvariable=tk.StringVar(value=lambda: f"{self.ocsvm_params['nu'].get():.2f}")).grid(row=1, column=2)
        
        ttk.Label(ocsvm_tab, text="Gamma:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        gamma_options = ['auto', 'scale']
        gamma_dropdown = ttk.Combobox(ocsvm_tab, textvariable=self.ocsvm_params['gamma'], values=gamma_options, state="readonly")
        gamma_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Isolation Forest parameters tab
        iforest_tab = ttk.Frame(param_notebook)
        param_notebook.add(iforest_tab, text="Isolation Forest")
        
        ttk.Label(iforest_tab, text="N Estimators:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        n_est_spinbox = ttk.Spinbox(iforest_tab, from_=10, to=500, textvariable=self.iforest_params['n_estimators'], width=10)
        n_est_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(iforest_tab, text="Contamination:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        contam_slider = ttk.Scale(iforest_tab, from_=0.01, to=0.5, variable=self.iforest_params['contamination'], orient=tk.HORIZONTAL)
        contam_slider.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(iforest_tab, textvariable=tk.StringVar(value=lambda: f"{self.iforest_params['contamination'].get():.2f}")).grid(row=1, column=2)
        
        ttk.Label(iforest_tab, text="Max Features:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        max_feat_slider = ttk.Scale(iforest_tab, from_=0.1, to=1.0, variable=self.iforest_params['max_features'], orient=tk.HORIZONTAL)
        max_feat_slider.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(iforest_tab, textvariable=tk.StringVar(value=lambda: f"{self.iforest_params['max_features'].get():.1f}")).grid(row=2, column=2)
        
        # LSTM parameters tab
        lstm_tab = ttk.Frame(param_notebook)
        param_notebook.add(lstm_tab, text="LSTM Network")
        
        ttk.Label(lstm_tab, text="First Layer Units:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        layer1_spinbox = ttk.Spinbox(lstm_tab, from_=16, to=256, textvariable=self.lstm_params['layer1_units'], width=10)
        layer1_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(lstm_tab, text="Second Layer Units:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        layer2_spinbox = ttk.Spinbox(lstm_tab, from_=8, to=128, textvariable=self.lstm_params['layer2_units'], width=10)
        layer2_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(lstm_tab, text="Dropout Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        dropout_slider = ttk.Scale(lstm_tab, from_=0.0, to=0.5, variable=self.lstm_params['dropout_rate'], orient=tk.HORIZONTAL)
        dropout_slider.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(lstm_tab, textvariable=tk.StringVar(value=lambda: f"{self.lstm_params['dropout_rate'].get():.2f}")).grid(row=2, column=2)
        
        ttk.Label(lstm_tab, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        lr_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        lr_dropdown = ttk.Combobox(lstm_tab, values=lr_values, state="readonly", width=10)
        lr_dropdown.set(0.001)
        lr_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        lr_dropdown.bind("<<ComboboxSelected>>", lambda e: self.lstm_params['learning_rate'].set(float(lr_dropdown.get())))
        
        ttk.Label(lstm_tab, text="Batch Size:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        batch_spinbox = ttk.Spinbox(lstm_tab, from_=8, to=64, textvariable=self.lstm_params['batch_size'], width=10)
        batch_spinbox.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(lstm_tab, text="Epochs:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        epochs_spinbox = ttk.Spinbox(lstm_tab, from_=10, to=500, textvariable=self.lstm_params['epochs'], width=10)
        epochs_spinbox.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Add buttons to save/load parameter configurations
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_params).pack(side=tk.RIGHT, padx=5)
    
    def filter_discharge_list(self, event=None):
        """Filter the discharge list based on user input"""
        typed = self.discharge_id_var.get().lower()
        if typed == '':
            self.update_discharge_list()  # If cleared, show all discharges
        else:
            # Filter the list based on what's typed
            filtered_ids = [
                discharge_id for discharge_id in self.all_discharge_ids
                if typed in discharge_id.lower()
            ]
            self.discharge_combobox['values'] = filtered_ids

    def update_discharge_list(self):
        """Update the discharge combobox with all available discharges"""
        if hasattr(self, 'classifier') and hasattr(self.classifier, 'discharge_labels'):
            self.all_discharge_ids = list(self.classifier.discharge_labels.keys())
            self.discharge_combobox['values'] = self.all_discharge_ids
            
            # Add visual indicators for training/test set
            if hasattr(self, 'training_ids') and self.training_ids:
                values_with_indicators = []
                for discharge_id in self.all_discharge_ids:
                    if discharge_id in self.training_ids:
                        values_with_indicators.append(f"{discharge_id} (Training)")
                    elif discharge_id in self.test_ids:
                        values_with_indicators.append(f"{discharge_id} (Test)")
                    else:
                        values_with_indicators.append(discharge_id)
                self.discharge_combobox['values'] = values_with_indicators

    def create_status_frame(self):
        # Progress bar
        self.progress = ttk.Progressbar(
            self.status_frame, 
            orient=tk.HORIZONTAL, 
            length=500, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            self.status_frame, 
            textvariable=self.status_var
        )
        self.status_label.pack(side=tk.RIGHT)
    
    def create_results_area(self):
        # Create notebook for tabbed results
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab for data information
        self.data_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.data_tab, text="Data Summary")
        
        self.data_text = scrolledtext.ScrolledText(self.data_tab)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab for model results
        self.models_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.models_tab, text="Model Performance")
        
        # Setup the frame for charts
        self.charts_frame = ttk.Frame(self.models_tab)
        self.charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tab for predictions
        self.predictions_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.predictions_tab, text="Predictions")
        
        self.prediction_text = scrolledtext.ScrolledText(self.predictions_tab)
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
    
    def update_step_indicators(self):
        for i, indicator in enumerate(self.step_indicators):
            if i < self.current_step:
                # Completed step
                indicator.config(text="✓", foreground="green")
            elif i == self.current_step:
                # Current step
                indicator.config(text="►", foreground="blue")
            else:
                # Future step
                indicator.config(text="○", foreground="black")
    
    def safe_update_ui(self, target_func, *args, **kwargs):
        """Thread-safe way to update UI elements"""
        if threading.current_thread() is threading.main_thread():
            # If we're already on the main thread, just call the function directly
            return target_func(*args, **kwargs)
        else:
            # Otherwise, schedule it to be run on the main thread
            return self.root.after(0, target_func, *args, **kwargs)

    def set_status(self, status_text):
        """Thread-safe status update"""
        self.safe_update_ui(self.status_var.set, status_text)

    def set_progress(self, progress_value):
        """Thread-safe progress update"""
        self.safe_update_ui(self.progress_var.set, progress_value)

    def run_with_progress(self, func, next_step=True):
        """Run a function with progress updates"""
        self.set_progress(0)
        
        # Disable all buttons during processing
        self.load_data_btn.config(state=tk.DISABLED)
        self.train_models_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
        
        def worker():
            try:
                func()
                
                # All UI updates need to be scheduled on the main thread
                def complete_task():
                    # First increment the step if needed
                    if next_step:
                        self.current_step += 1
                        self.update_step_indicators()
                    
                    # Then enable appropriate buttons based on the updated step
                    if self.current_step >= 1:
                        self.load_data_btn.config(state=tk.NORMAL)
                        self.train_models_btn.config(state=tk.NORMAL)
                    if self.current_step >= 2:
                        self.analyze_btn.config(state=tk.NORMAL)
                    if self.current_step >= 3:
                        self.predict_btn.config(state=tk.NORMAL)
                    
                    self.progress_var.set(100)
                
                self.safe_update_ui(complete_task)
                
            except Exception as e:
                def show_error():
                    messagebox.showerror("Error", str(e))
                    self.status_var.set(f"Error: {str(e)}")
                    self.load_data_btn.config(state=tk.NORMAL)
                
                self.safe_update_ui(show_error)
        
        threading.Thread(target=worker).start()
    
    def load_data(self):
        self.set_status("Loading discharge data...")
        
        def process():
            progress_steps = 3
            
            # Step 1: Load discharge list
            self.set_progress(100/progress_steps * 0)
            discharge_df = self.classifier.load_discharge_list()
            
            # Step 2: Load all data
            self.set_progress(100/progress_steps * 1)
            self.set_status("Loading feature data...")
            self.classifier.load_all_data()
            
            # Step 3: Update UI with data summary
            self.set_progress(100/progress_steps * 2)
            self.set_status("Processing data summary...")
            
            # Create data summary
            total_discharges = len(self.classifier.discharge_labels)
            disruptive = sum(1 for label in self.classifier.discharge_labels.values() if label == 1)
            non_disruptive = total_discharges - disruptive
            
            summary = f"Data Summary\n{'-'*50}\n"
            summary += f"Total discharges: {total_discharges}\n"
            summary += f"Disruptive discharges: {disruptive} ({disruptive/total_discharges*100:.1f}%)\n"
            summary += f"Non-disruptive discharges: {non_disruptive} ({non_disruptive/total_discharges*100:.1f}%)\n\n"
            summary += "Discharge IDs:\n"
            
            for i, (discharge_id, label) in enumerate(self.classifier.discharge_labels.items()):
                summary += f"{discharge_id} ({'Disruptive' if label == 1 else 'Non-disruptive'})"
                if i % 3 == 2:
                    summary += "\n"
                else:
                    summary += "\t"
            
            # Schedule UI updates on the main thread
            def update_text():
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, summary)
                self.set_status(f"Data loaded: {total_discharges} discharges")
                
                # Update discharge list in the combobox
                self.update_discharge_list()
            
            self.safe_update_ui(update_text)
        
        self.run_with_progress(process)
    
    def train_models(self):
        self.set_status("Training models with custom parameters...")
        
        def process():
            test_size = self.test_size_var.get()
            
            # Clear previous results (schedule on main thread)
            def clear_charts():
                for widget in self.charts_frame.winfo_children():
                    widget.destroy()
            
            self.safe_update_ui(clear_charts)
                
            # Train models
            self.set_progress(10)
            self.set_status("Preparing features...")
            time.sleep(0.3)
            
            # Verify parameters were properly initialized
            if not hasattr(self, 'ocsvm_params') or not hasattr(self, 'iforest_params') or not hasattr(self, 'lstm_params'):
                self.init_model_parameters()  # Reinitialize if missing
            
            # Get current parameter values - with additional logging
            try:
                ocsvm_settings = {
                    'kernel': self.ocsvm_params['kernel'].get(),
                    'nu': float(self.ocsvm_params['nu'].get()),  # Ensure float conversion
                    'gamma': self.ocsvm_params['gamma'].get()
                }
                
                iforest_settings = {
                    'n_estimators': int(self.iforest_params['n_estimators'].get()),  # Ensure int conversion
                    'contamination': float(self.iforest_params['contamination'].get()),  # Ensure float conversion
                    'max_features': float(self.iforest_params['max_features'].get())  # Ensure float conversion
                }
                
                lstm_settings = {
                    'layer1_units': int(self.lstm_params['layer1_units'].get()),  # Ensure int conversion
                    'layer2_units': int(self.lstm_params['layer2_units'].get()),  # Ensure int conversion
                    'dropout_rate': float(self.lstm_params['dropout_rate'].get()),  # Ensure float conversion
                    'learning_rate': float(self.lstm_params['learning_rate'].get()),  # Ensure float conversion
                    'batch_size': int(self.lstm_params['batch_size'].get()),  # Ensure int conversion
                    'epochs': int(self.lstm_params['epochs'].get())  # Ensure int conversion
                }
                
                # Print parameter values to console for debugging
                print("\nParameters being sent from GUI:")
                print(f"OCSVM: {ocsvm_settings}")
                print(f"Isolation Forest: {iforest_settings}")
                print(f"LSTM: {lstm_settings}")
                print(f"Test size: {test_size}")
                
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                # More specific error handling with detailed logging
                print(f"Error retrieving parameters: {str(e)}")
                self.set_status(f"Warning: Using default parameters due to {str(e)}")
                ocsvm_settings = None
                iforest_settings = None
                lstm_settings = None
            
            self.set_progress(20)
            self.set_status(f"Training OCSVM model with kernel={ocsvm_settings['kernel']}, nu={ocsvm_settings['nu']:.2f}...")
            time.sleep(0.3)
            
            self.set_progress(40)
            self.set_status(f"Training Isolation Forest with n_estimators={iforest_settings['n_estimators']}...")
            time.sleep(0.3)
            
            self.set_progress(60)
            self.set_status(f"Training LSTM network with {lstm_settings['layer1_units']} units in layer 1...")
            
            # Actually train the models with custom parameters
            self.results = self.classifier.train_models(
                test_size=test_size,
                ocsvm_params=ocsvm_settings,
                iforest_params=iforest_settings,
                lstm_params=lstm_settings
            )
            
            # Store training and test discharge IDs
            self.test_ids = set(self.results['ids'])
            self.training_ids = set(self.classifier.discharge_labels.keys()) - self.test_ids
            
            self.set_progress(80)
            self.set_status("Generating performance metrics...")
            time.sleep(0.3)
            
            # Create figures
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            models = ['OCSVM', 'Isolation Forest', 'LSTM', 'Combined']
            
            accuracies = []
            for model_name in ['ocsvm', 'iforest', 'lstm', 'combined']:
                acc = np.sum(self.results[model_name] == self.results['true']) / len(self.results['true'])
                accuracies.append(acc)
            
            ax1.bar(models, accuracies, color=['lightblue', 'lightgreen', 'coral', 'gold'])
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylim([0, 1])
            
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
            # Create confusion matrices
            fig2, ax2 = plt.subplots(2, 2, figsize=(8, 8))
            fig2.suptitle('Confusion Matrices')
            
            models_list = ['ocsvm', 'iforest', 'lstm', 'combined']
            titles = ['One-Class SVM', 'Isolation Forest', 'LSTM', 'Combined']
            
            for i, (model, title) in enumerate(zip(models_list, titles)):
                cm = confusion_matrix(self.results['true'], self.results[model])
                row, col = i // 2, i % 2
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2[row, col])
                ax2[row, col].set_title(title)
                ax2[row, col].set_xlabel('Predicted')
                ax2[row, col].set_ylabel('Actual')
            
            fig2.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Update UI on main thread
            def update_charts():
                canvas1 = FigureCanvasTkAgg(fig1, master=self.charts_frame)
                canvas1.draw()
                canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                canvas2 = FigureCanvasTkAgg(fig2, master=self.charts_frame)
                canvas2.draw()
                canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                # Update the discharge list to show training/test indicators
                self.update_discharge_list()
                
                self.set_status("Models trained successfully")
                
                # Make sure we reach step 3 after training (fix for predict button not being enabled)
                self.current_step = 3
            
            self.safe_update_ui(update_charts)
        
        self.run_with_progress(process)
    
    def is_training_discharge(self, discharge_id):
        """Check if a discharge ID was used in the training set"""
        return discharge_id in self.training_ids
    
    def is_test_discharge(self, discharge_id):
        """Check if a discharge ID was used in the test set"""
        return discharge_id in self.test_ids
    
    def analyze_results(self):
        self.status_var.set("Analyzing test set results...")
        
        def process():
            if not hasattr(self.classifier, 'test_data'):
                messagebox.showerror("Error", "No test data available. Run training first.")
                return
            
            # Create stdout redirect to capture prints
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                self.classifier.analyze_test_set()
            
            analysis_results = output.getvalue()
            
            # Display in predictions tab
            self.prediction_text.delete(1.0, tk.END)
            self.prediction_text.insert(tk.END, "Test Set Analysis\n")
            self.prediction_text.insert(tk.END, "="*50 + "\n\n")
            self.prediction_text.insert(tk.END, analysis_results)
            
            # Switch to the predictions tab
            self.results_notebook.select(self.predictions_tab)
            
            self.status_var.set("Analysis complete")
        
        self.run_with_progress(process, next_step=False)
    
    def predict_discharge(self):
        # Extract just the discharge ID from the combobox value
        # which might include "(Training)" or "(Test)" indicators
        combobox_value = self.discharge_id_var.get()
        discharge_id = combobox_value.split(" (")[0].strip()
        
        if not discharge_id:
            messagebox.showwarning("Input Error", "Please select a discharge ID")
            return
        
        self.set_status(f"Predicting for discharge {discharge_id}...")
        
        def process():
            try:
                # Create stdout redirect to capture prints
                import io
                from contextlib import redirect_stdout
                
                output = io.StringIO()
                with redirect_stdout(output):
                    result = self.classifier.predict_discharge(discharge_id)
                
                prediction_results = output.getvalue()
                
                if result:
                    # Get the voting results
                    models = ['OCSVM', 'Isolation Forest', 'LSTM']
                    predictions = [result['ocsvm'], result['iforest'], result['lstm']]
                    actual = result['actual']
                    
                    # Check if discharge was in training data
                    is_training = self.is_training_discharge(discharge_id)
                    is_test = self.is_test_discharge(discharge_id)
                    data_status = "TRAINING SET" if is_training else ("TEST SET" if is_test else "NEW DATA")
                    
                    # Create a figure to visualize the voting
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Voting results
                    ax1.bar(models, predictions, color=['lightblue', 'lightgreen', 'coral'])
                    ax1.set_ylim([0, 1.2])
                    ax1.set_ylabel('Prediction (1=Disruptive)')
                    ax1.set_title(f'Model Predictions for Discharge {discharge_id}')
                    
                    for i, v in enumerate(predictions):
                        ax1.text(i, v + 0.05, str(v), ha='center')
                    
                    # Majority vote visualization
                    labels = 'Disruptive Votes', 'Non-disruptive Votes'
                    sizes = [sum(predictions), 3-sum(predictions)]
                    colors = ['red', 'blue']
                    explode = (0.1, 0)
                    
                    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                            autopct='%1.1f%%', shadow=True, startangle=90)
                    ax2.axis('equal')
                    ax2.set_title('Voting Distribution')
                    
                    plt.tight_layout()
                    
                    # Schedule UI updates on main thread
                    def update_ui():
                        # Display in the predictions tab
                        for widget in self.predictions_tab.winfo_children():
                            widget.destroy()
                        
                        # Add text area
                        self.prediction_text = scrolledtext.ScrolledText(self.predictions_tab, height=10)
                        self.prediction_text.pack(fill=tk.X)
                        
                        # Add prediction text
                        self.prediction_text.insert(tk.END, prediction_results)
                        
                        # Add data source warning if needed
                        if is_training:
                            warning_frame = ttk.Frame(self.predictions_tab)
                            warning_frame.pack(fill=tk.X, pady=5)
                            
                            warning_label = ttk.Label(
                                warning_frame,
                                text="⚠️ WARNING: This discharge was used for TRAINING the model! ⚠️",
                                font=("Arial", 12, "bold"),
                                foreground="red",
                                background="yellow"
                            )
                            warning_label.pack(fill=tk.X, padx=10, pady=5)
                            
                            warning_text = ttk.Label(
                                warning_frame,
                                text="Results may be biased since the model has seen this data before.",
                                font=("Arial", 10),
                                foreground="black"
                            )
                            warning_text.pack(fill=tk.X, padx=10, pady=5)
                        
                        # Add canvas
                        canvas = FigureCanvasTkAgg(fig, master=self.predictions_tab)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        
                        # Add final result label with large colored text
                        result_frame = ttk.Frame(self.predictions_tab)
                        result_frame.pack(fill=tk.X, pady=10)
                        
                        # Show data source
                        ttk.Label(
                            result_frame,
                            text=f"DATA SOURCE: {data_status}",
                            font=("Arial", 10),
                            foreground="blue" if is_test else ("red" if is_training else "purple")
                        ).pack()
                        
                        final_result = "DISRUPTIVE" if result['final'] == 1 else "NON-DISRUPTIVE"
                        result_color = "red" if result['final'] == 1 else "green"
                        
                        ttk.Label(
                            result_frame, 
                            text=f"FINAL PREDICTION: {final_result}",
                            font=("Arial", 16, "bold"),
                            foreground=result_color
                        ).pack()
                        
                        actual_result = "DISRUPTIVE" if result['actual'] == 1 else "NON-DISRUPTIVE"
                        actual_color = "red" if result['actual'] == 1 else "green"
                        
                        ttk.Label(
                            result_frame, 
                            text=f"ACTUAL VALUE: {actual_result}",
                            font=("Arial", 14),
                            foreground=actual_color
                        ).pack()
                        
                        # Show if prediction was correct
                        correct = result['final'] == result['actual']
                        ttk.Label(
                            result_frame, 
                            text=f"PREDICTION WAS {'CORRECT' if correct else 'INCORRECT'}",
                            font=("Arial", 12, "bold"),
                            foreground="blue" if correct else "purple"
                        ).pack()
                        
                        # Switch to the predictions tab
                        self.results_notebook.select(self.predictions_tab)
                    
                    self.safe_update_ui(update_ui)
                
                self.set_status(f"Prediction for discharge {discharge_id} complete")
                
            except Exception as e:
                def show_error():
                    messagebox.showerror("Prediction Error", str(e))
                    self.status_var.set(f"Error: {str(e)}")
                
                self.safe_update_ui(show_error)
        
        self.run_with_progress(process, next_step=False)
    
    def save_config(self):
        """Save current model parameters to a JSON file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Model Configuration"
            )
            
            if not file_path:
                return  # User cancelled
                
            config = {
                'ocsvm': {
                    'kernel': self.ocsvm_params['kernel'].get(),
                    'nu': self.ocsvm_params['nu'].get(),
                    'gamma': self.ocsvm_params['gamma'].get()
                },
                'iforest': {
                    'n_estimators': self.iforest_params['n_estimators'].get(),
                    'contamination': self.iforest_params['contamination'].get(),
                    'max_features': self.iforest_params['max_features'].get()
                },
                'lstm': {
                    'layer1_units': self.lstm_params['layer1_units'].get(),
                    'layer2_units': self.lstm_params['layer2_units'].get(),
                    'dropout_rate': self.lstm_params['dropout_rate'].get(),
                    'learning_rate': self.lstm_params['learning_rate'].get(),
                    'batch_size': self.lstm_params['batch_size'].get(),
                    'epochs': self.lstm_params['epochs'].get()
                },
                'test_size': self.test_size_var.get()
            }
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.status_var.set(f"Configuration saved to {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load model parameters from a JSON file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Model Configuration"
            )
            
            if not file_path:
                return  # User cancelled
                
            with open(file_path, 'r') as f:
                config = json.load(f)
                
            # Update OCSVM parameters
            if 'ocsvm' in config:
                for param, value in config['ocsvm'].items():
                    if param in self.ocsvm_params:
                        self.ocsvm_params[param].set(value)
            
            # Update Isolation Forest parameters
            if 'iforest' in config:
                for param, value in config['iforest'].items():
                    if param in self.iforest_params:
                        self.iforest_params[param].set(value)
            
            # Update LSTM parameters
            if 'lstm' in config:
                for param, value in config['lstm'].items():
                    if param in self.lstm_params:
                        self.lstm_params[param].set(value)
            
            # Update test size
            if 'test_size' in config:
                self.test_size_var.set(config['test_size'])
                
            self.status_var.set(f"Configuration loaded from {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")
    
    def reset_params(self):
        """Reset parameters to their default values"""
        # Reset OCSVM params
        self.ocsvm_params['kernel'].set('rbf')
        self.ocsvm_params['nu'].set(0.1)
        self.ocsvm_params['gamma'].set('auto')
        
        # Reset Isolation Forest params
        self.iforest_params['n_estimators'].set(100)
        self.iforest_params['contamination'].set(0.1)
        self.iforest_params['max_features'].set(1.0)
        
        # Reset LSTM params
        self.lstm_params['layer1_units'].set(64)
        self.lstm_params['layer2_units'].set(32)
        self.lstm_params['dropout_rate'].set(0.2)
        self.lstm_params['learning_rate'].set(0.001)
        self.lstm_params['batch_size'].set(16)
        self.lstm_params['epochs'].set(100)
        
        # Reset test size
        self.test_size_var.set(0.3)
        
        self.status_var.set("Parameters reset to default values")

if __name__ == "__main__":
    root = tk.Tk()
    app = DisruptionClassifierGUI(root)
    root.update()  # For proper initial sizing
    
    # Add a styled intro message
    intro_msg = """
    Disruption Classifier GUI
    
    This application provides a visual interface to the plasma disruption classifier.
    Follow the workflow steps on the left to:
    
    1. Load discharge data
    2. Train the machine learning models
    3. Analyze the test results
    4. Make predictions on individual discharges
    
    The application implements three machine learning models:
    - One-Class SVM
    - Isolation Forest
    - LSTM Neural Network
    
    And combines them with a majority voting system.
    """
    
    app.data_text.insert(tk.END, intro_msg)
    
    root.mainloop()

