import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import itertools
import time
import tkinter.simpledialog as simpledialog

# Import the DisruptionClassifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from classifier import DisruptionClassifier, DISCHARGE_DATA_PATH, DISCHARGES_ETIQUETATION_DATA_FILE

class AutotuneConfigDialog(tk.Toplevel):
    """Dialog for configuring autotune parameter ranges"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Autotune Configuration")
        self.geometry("550x400")
        self.resizable(False, False)
        
        self.params = {
            "n_estimators": {"start": 50, "end": 300, "step": 50},
            "contamination": {"start": 0.05, "end": 0.3, "step": 0.05},
            "max_features": {"start": 0.5, "end": 1.0, "step": 0.1},
            "max_samples": {"start": 0.5, "end": 1.0, "step": 0.1}
        }
        
        self.result = None
        self.create_widgets()
        
        # Make this dialog modal
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_widgets(self):
        # Create the main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create labels
        ttk.Label(main_frame, text="Parameter", font=("", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Start", font=("", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="End", font=("", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(main_frame, text="Step", font=("", 10, "bold")).grid(row=0, column=3, padx=5, pady=5)
        
        # Store variables
        self.var_dict = {}
        for i, (param_name, values) in enumerate(self.params.items(), 1):
            ttk.Label(main_frame, text=param_name).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            
            # Create dictionary to store variables
            self.var_dict[param_name] = {
                "start": tk.DoubleVar(value=values["start"]),
                "end": tk.DoubleVar(value=values["end"]),
                "step": tk.DoubleVar(value=values["step"])
            }
            
            # Create entry fields
            ttk.Entry(main_frame, textvariable=self.var_dict[param_name]["start"], width=8).grid(
                row=i, column=1, padx=5, pady=5)
            ttk.Entry(main_frame, textvariable=self.var_dict[param_name]["end"], width=8).grid(
                row=i, column=2, padx=5, pady=5)
            ttk.Entry(main_frame, textvariable=self.var_dict[param_name]["step"], width=8).grid(
                row=i, column=3, padx=5, pady=5)
        
        # Create a frame for the combination count
        count_frame = ttk.LabelFrame(main_frame, text="Combinations")
        count_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=10, sticky="we")
        
        self.combinations_label = ttk.Label(count_frame, text="Total configurations to test: 0")
        self.combinations_label.pack(pady=5)
        
        ttk.Button(count_frame, text="Calculate Combinations", command=self.calculate_combinations).pack(pady=5)
        
        # Create buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=6, column=0, columnspan=4, pady=10)
        
        ttk.Button(btn_frame, text="OK", command=self.on_ok).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).grid(row=0, column=1, padx=10)
        
    def calculate_combinations(self):
        """Calculate and display the total number of combinations"""
        try:
            total = 1
            for param_name in self.params:
                start = self.var_dict[param_name]["start"].get()
                end = self.var_dict[param_name]["end"].get()
                step = self.var_dict[param_name]["step"].get()
                
                # Handle the special case of integers like n_estimators
                if param_name == "n_estimators":
                    values = list(range(int(start), int(end) + int(step), int(step)))
                else:
                    # Calculate number of steps for floating point
                    steps = int(round((end - start) / step)) + 1
                    values = [round(start + i * step, 3) for i in range(steps)]
                    # Make sure we don't exceed the end value
                    values = [v for v in values if v <= end]
                
                total *= len(values)
            
            self.combinations_label.config(text=f"Total configurations to test: {total}")
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating combinations: {str(e)}")
            
    def on_ok(self):
        """Save the parameters and close the dialog"""
        try:
            # Convert entries to actual values
            result = {}
            for param_name in self.params:
                start = self.var_dict[param_name]["start"].get()
                end = self.var_dict[param_name]["end"].get()
                step = self.var_dict[param_name]["step"].get()
                
                if param_name == "n_estimators":
                    # Integer parameter
                    values = list(range(int(start), int(end) + int(step), int(step)))
                else:
                    # Float parameter
                    steps = int(round((end - start) / step)) + 1
                    values = [round(start + i * step, 3) for i in range(steps)]
                    values = [v for v in values if v <= end]
                
                result[param_name] = values
            
            self.result = result
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def on_cancel(self):
        """Cancel and close the dialog"""
        self.result = None
        self.destroy()

class DischargeSelectionDialog(tk.Toplevel):
    """Dialog for selecting discharges for training and testing"""
    def __init__(self, parent, discharge_ids, current_selection=None):
        super().__init__(parent)
        self.parent = parent
        self.discharge_ids = sorted(discharge_ids)
        self.current_selection = current_selection or {"train": [], "test": []}
        
        self.title("Select Discharges")
        self.geometry("800x500")
        
        self.result = None
        self.create_widgets()
        
        # Make this dialog modal
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_widgets(self):
        # Create main frames
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        all_frame = ttk.LabelFrame(main_frame, text="Available Discharges")
        all_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ns")
        
        train_frame = ttk.LabelFrame(main_frame, text="Training Set")
        train_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        test_frame = ttk.LabelFrame(main_frame, text="Test Set")
        test_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_columnconfigure(3, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create listboxes with scrollbars
        self.all_listbox = tk.Listbox(all_frame, selectmode=tk.MULTIPLE)
        self.all_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        all_scrollbar = ttk.Scrollbar(all_frame, orient=tk.VERTICAL, command=self.all_listbox.yview)
        all_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.all_listbox.config(yscrollcommand=all_scrollbar.set)
        
        self.train_listbox = tk.Listbox(train_frame, selectmode=tk.MULTIPLE)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        train_scrollbar = ttk.Scrollbar(train_frame, orient=tk.VERTICAL, command=self.train_listbox.yview)
        train_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_listbox.config(yscrollcommand=train_scrollbar.set)
        
        self.test_listbox = tk.Listbox(test_frame, selectmode=tk.MULTIPLE)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        test_scrollbar = ttk.Scrollbar(test_frame, orient=tk.VERTICAL, command=self.test_listbox.yview)
        test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_listbox.config(yscrollcommand=test_scrollbar.set)
        
        # Populate listboxes
        self.populate_listboxes()
        
        # Create buttons
        ttk.Button(button_frame, text="→ Train", command=self.move_to_train).pack(pady=5)
        ttk.Button(button_frame, text="→ Test", command=self.move_to_test).pack(pady=5)
        ttk.Button(button_frame, text="← Remove", command=self.move_to_available).pack(pady=20)
        ttk.Button(button_frame, text="Auto Split", command=self.auto_split).pack(pady=5)
        
        # Stats frame
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="we")
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.pack(pady=5)
        self.update_stats()
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="we")
        
        ttk.Button(action_frame, text="Apply Selection", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=5)
        
    def populate_listboxes(self):
        """Populate the listboxes with discharge IDs"""
        # Clear listboxes
        self.all_listbox.delete(0, tk.END)
        self.train_listbox.delete(0, tk.END)
        self.test_listbox.delete(0, tk.END)
        
        # Add discharges to the appropriate listboxes
        all_ids = set(self.discharge_ids)
        train_ids = set(self.current_selection["train"])
        test_ids = set(self.current_selection["test"])
        
        # Available are those not in train or test
        available_ids = all_ids - train_ids - test_ids
        
        for discharge_id in sorted(available_ids):
            self.all_listbox.insert(tk.END, discharge_id)
        
        for discharge_id in sorted(train_ids):
            self.train_listbox.insert(tk.END, discharge_id)
            
        for discharge_id in sorted(test_ids):
            self.test_listbox.insert(tk.END, discharge_id)
    
    def update_stats(self):
        """Update the statistics label"""
        total = len(self.discharge_ids)
        train = self.train_listbox.size()
        test = self.test_listbox.size()
        available = self.all_listbox.size()
        
        self.stats_label.config(
            text=f"Total: {total} discharges | Training: {train} ({train/total*100:.1f}%) | "
                 f"Testing: {test} ({test/total*100:.1f}%) | Available: {available} ({available/total*100:.1f}%)"
        )
    
    def move_to_train(self):
        """Move selected discharges from available to training set"""
        selected_indices = self.all_listbox.curselection()
        if not selected_indices:
            return
            
        # Get the selected discharge IDs
        selected_ids = [self.all_listbox.get(i) for i in selected_indices]
        
        # Add to train listbox
        for discharge_id in selected_ids:
            self.train_listbox.insert(tk.END, discharge_id)
        
        # Remove from all listbox (need to do it in reverse order to avoid index shifts)
        for index in sorted(selected_indices, reverse=True):
            self.all_listbox.delete(index)
            
        self.update_stats()
    
    def move_to_test(self):
        """Move selected discharges from available to test set"""
        selected_indices = self.all_listbox.curselection()
        if not selected_indices:
            return
            
        # Get the selected discharge IDs
        selected_ids = [self.all_listbox.get(i) for i in selected_indices]
        
        # Add to test listbox
        for discharge_id in selected_ids:
            self.test_listbox.insert(tk.END, discharge_id)
        
        # Remove from all listbox (need to do it in reverse order to avoid index shifts)
        for index in sorted(selected_indices, reverse=True):
            self.all_listbox.delete(index)
            
        self.update_stats()
    
    def move_to_available(self):
        """Move selected discharges back to available"""
        # Handle train selections
        train_indices = self.train_listbox.curselection()
        if train_indices:
            # Get the selected discharge IDs
            selected_ids = [self.train_listbox.get(i) for i in train_indices]
            
            # Add to available listbox
            for discharge_id in selected_ids:
                self.all_listbox.insert(tk.END, discharge_id)
            
            # Remove from train listbox (need to do it in reverse order to avoid index shifts)
            for index in sorted(train_indices, reverse=True):
                self.train_listbox.delete(index)
        
        # Handle test selections
        test_indices = self.test_listbox.curselection()
        if test_indices:
            # Get the selected discharge IDs
            selected_ids = [self.test_listbox.get(i) for i in test_indices]
            
            # Add to available listbox
            for discharge_id in selected_ids:
                self.all_listbox.insert(tk.END, discharge_id)
            
            # Remove from test listbox (need to do it in reverse order to avoid index shifts)
            for index in sorted(test_indices, reverse=True):
                self.test_listbox.delete(index)
                
        # Sort the available listbox
        temp = list(self.all_listbox.get(0, tk.END))
        temp.sort()
        self.all_listbox.delete(0, tk.END)
        for item in temp:
            self.all_listbox.insert(tk.END, item)
                
        self.update_stats()
    
    def auto_split(self):
        """Automatically split discharges into train and test sets"""
        # Get all discharge IDs
        all_ids = list(self.all_listbox.get(0, tk.END))
        train_ids = list(self.train_listbox.get(0, tk.END))
        test_ids = list(self.test_listbox.get(0, tk.END))
        
        # Combine all discharges
        all_discharges = all_ids + train_ids + test_ids
        
        # Prompt for test percentage
        test_pct = simpledialog.askfloat(
            "Test Set Size", 
            "Enter the percentage for test set (e.g., 30 for 30%):",
            minvalue=1, maxvalue=99, initialvalue=30
        )
        
        if test_pct is None:
            return  # User canceled
            
        # Convert to decimal
        test_size = test_pct / 100
        
        # Shuffle and split
        np.random.shuffle(all_discharges)
        split_index = int((1 - test_size) * len(all_discharges))
        
        new_train = all_discharges[:split_index]
        new_test = all_discharges[split_index:]
        
        # Update listboxes
        self.all_listbox.delete(0, tk.END)
        self.train_listbox.delete(0, tk.END)
        self.test_listbox.delete(0, tk.END)
        
        for item in sorted(new_train):
            self.train_listbox.insert(tk.END, item)
            
        for item in sorted(new_test):
            self.test_listbox.insert(tk.END, item)
            
        self.update_stats()
    
    def on_ok(self):
        """Save the selection and close the dialog"""
        train_ids = list(self.train_listbox.get(0, tk.END))
        test_ids = list(self.test_listbox.get(0, tk.END))
        
        # Check if both sets have data
        if not train_ids or not test_ids:
            messagebox.showerror("Error", "Both training and test sets must contain discharges.")
            return
        
        self.result = {
            "train": train_ids,
            "test": test_ids
        }
        self.destroy()
    
    def on_cancel(self):
        """Cancel and close the dialog"""
        self.result = None
        self.destroy()

class IForestTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Isolation Forest Parameter Tuner")
        self.root.geometry("1200x800")
        
        # Selected discharges
        self.selected_discharges = None
        
        # Load data
        self.load_data()
        
        # Create GUI components
        self.create_parameter_frame()
        self.create_results_frame()
        self.create_visualization_frame()
        
        # Store best parameters and results
        self.best_params = None
        self.best_score = -1  # Using F1 score as metric to maximize
        self.test_results = {}
        self.train_results = {}
        
    def load_data(self):
        """Load and prepare data for training"""
        print("Loading data...")
        self.results_text = tk.Text(self.root, width=60, height=20)
        self.results_text.grid(row=0, column=1, padx=10, pady=10)
        self.results_text.insert(tk.END, "Loading and preparing data...\n")
        
        self.classifier = DisruptionClassifier(DISCHARGE_DATA_PATH, DISCHARGES_ETIQUETATION_DATA_FILE)
        self.classifier.load_all_data()
        
        # Prepare features
        X_scaled, y, discharge_ids = self.classifier.prepare_features()
        
        # Print dataset information
        self.results_text.insert(tk.END, f"\n=== Dataset Information ===\n")
        self.results_text.insert(tk.END, f"Total samples: {len(X_scaled)}\n")
        self.results_text.insert(tk.END, f"Features per sample: {X_scaled.shape[1]}\n")
        self.results_text.insert(tk.END, f"Class distribution: {np.sum(y == 0)} non-disruptive, {np.sum(y == 1)} disruptive\n")
        self.results_text.insert(tk.END, f"Class imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}:1\n")
        
        # Split data (using the same test_size as in the conf.json)
        from sklearn.model_selection import train_test_split
        try:
            with open("c:\\Users\\Ruben\\Desktop\\conf.json", 'r') as f:
                conf = json.load(f)
                test_size = conf.get("test_size", 0.3)
                self.results_text.insert(tk.END, f"Using test_size={test_size} from conf.json\n")
        except:
            test_size = 0.3
            self.results_text.insert(tk.END, f"Using default test_size={test_size}\n")
            
        self.X_train, self.X_test, self.y_train, self.y_test, ids_train, ids_test = train_test_split(
            X_scaled, y, discharge_ids, test_size=test_size, random_state=42, stratify=y
        )
        
        # Extract normal data for training (as in original code)
        self.X_train_normal = self.X_train[self.y_train == 0]
        self.X_train_disruptive = self.X_train[self.y_train == 1]
        
        # Store discharge IDs for reference
        self.ids_train = ids_train
        self.ids_test = ids_test
        
        # Dataset statistics
        self.results_text.insert(tk.END, f"\n=== Training/Test Split ===\n")
        self.results_text.insert(tk.END, f"Training set: {len(self.X_train)} samples "
                            f"({np.sum(self.y_train == 0)} non-disruptive, {np.sum(self.y_train == 1)} disruptive)\n")
        self.results_text.insert(tk.END, f"Test set: {len(self.X_test)} samples "
                            f"({np.sum(self.y_test == 0)} non-disruptive, {np.sum(self.y_test == 1)} disruptive)\n")
        
        # Feature statistics
        self.results_text.insert(tk.END, f"\n=== Feature Statistics ===\n")
        self.results_text.insert(tk.END, f"Mean of normal samples: {np.mean(self.X_train_normal, axis=0)[:3]}...\n")
        self.results_text.insert(tk.END, f"Std of normal samples: {np.std(self.X_train_normal, axis=0)[:3]}...\n")
        
        if len(self.X_train_disruptive) > 0:
            self.results_text.insert(tk.END, f"Mean of disruptive samples: {np.mean(self.X_train_disruptive, axis=0)[:3]}...\n")
            self.results_text.insert(tk.END, f"Std of disruptive samples: {np.std(self.X_train_disruptive, axis=0)[:3]}...\n")
        
        print(f"Data loaded. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        print(f"Normal samples for training: {self.X_train_normal.shape[0]}")
        print(f"Disruptive samples for training: {self.X_train_disruptive.shape[0]}")
        
        # Remove this temporary text widget
        self.results_text.grid_forget()
        self.results_text = None
        
    def create_parameter_frame(self):
        """Create the frame for parameter inputs"""
        param_frame = ttk.LabelFrame(self.root, text="Isolation Forest Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # n_estimators parameter
        ttk.Label(param_frame, text="n_estimators:").grid(row=0, column=0, padx=5, pady=5)
        self.n_estimators_var = tk.IntVar(value=100)
        n_estimators_scale = ttk.Scale(
            param_frame, from_=10, to=500, orient=tk.HORIZONTAL, 
            variable=self.n_estimators_var, length=200
        )
        n_estimators_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(param_frame, textvariable=self.n_estimators_var).grid(row=0, column=2, padx=5, pady=5)
        
        # contamination parameter
        ttk.Label(param_frame, text="contamination:").grid(row=1, column=0, padx=5, pady=5)
        self.contamination_var = tk.DoubleVar(value=0.1)
        contamination_scale = ttk.Scale(
            param_frame, from_=0.01, to=0.5, orient=tk.HORIZONTAL, 
            variable=self.contamination_var, length=200
        )
        contamination_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(param_frame, textvariable=self.contamination_var).grid(row=1, column=2, padx=5, pady=5)
        
        # max_features parameter
        ttk.Label(param_frame, text="max_features:").grid(row=2, column=0, padx=5, pady=5)
        self.max_features_var = tk.DoubleVar(value=1.0)
        max_features_scale = ttk.Scale(
            param_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
            variable=self.max_features_var, length=200
        )
        max_features_scale.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(param_frame, textvariable=self.max_features_var).grid(row=2, column=2, padx=5, pady=5)
        
        # max_samples parameter (new parameter to test)
        ttk.Label(param_frame, text="max_samples:").grid(row=3, column=0, padx=5, pady=5)
        self.max_samples_var = tk.DoubleVar(value=0.5)
        max_samples_scale = ttk.Scale(
            param_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
            variable=self.max_samples_var, length=200
        )
        max_samples_scale.grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(param_frame, textvariable=self.max_samples_var).grid(row=3, column=2, padx=5, pady=5)
        
        # Training options
        train_frame = ttk.LabelFrame(param_frame, text="Training Options")
        train_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        
        self.normal_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            train_frame, text="Train on normal data only", 
            variable=self.normal_only_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Buttons
        button_frame = ttk.Frame(param_frame)
        button_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Train & Test", command=self.train_and_test).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Save Best Config", command=self.save_best_config).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset Best", command=self.reset_best).grid(row=0, column=2, padx=5)
        
        # Data selection and autotune buttons
        ttk.Button(
            param_frame, 
            text="Select Discharges", 
            command=self.select_discharges
        ).grid(row=6, column=0, columnspan=3, padx=5, pady=10, sticky="we")
        
        ttk.Button(
            param_frame, 
            text="Autotune Parameters", 
            command=self.open_autotune_dialog
        ).grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        
    def create_results_frame(self):
        """Create the frame for displaying results"""
        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")
        
        # Create a text widget to display results
        self.results_text = tk.Text(results_frame, width=60, height=20)
        self.results_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Display initial dataset information
        self.show_dataset_info()
        
    def show_dataset_info(self):
        """Show detailed information about the dataset"""
        self.results_text.insert(tk.END, "=== Dataset Information ===\n")
        self.results_text.insert(tk.END, f"Training set: {len(self.X_train)} samples\n")
        self.results_text.insert(tk.END, f"  - Non-disruptive: {np.sum(self.y_train == 0)} ({np.sum(self.y_train == 0)/len(self.y_train)*100:.1f}%)\n")
        self.results_text.insert(tk.END, f"  - Disruptive: {np.sum(self.y_train == 1)} ({np.sum(self.y_train == 1)/len(self.y_train)*100:.1f}%)\n")
        self.results_text.insert(tk.END, f"Test set: {len(self.X_test)} samples\n")
        self.results_text.insert(tk.END, f"  - Non-disruptive: {np.sum(self.y_test == 0)} ({np.sum(self.y_test == 0)/len(self.y_test)*100:.1f}%)\n")
        self.results_text.insert(tk.END, f"  - Disruptive: {np.sum(self.y_test == 1)} ({np.sum(self.y_test == 1)/len(self.y_test)*100:.1f}%)\n")
        self.results_text.insert(tk.END, f"\nTraining on normal data only will use {self.X_train_normal.shape[0]} samples\n")
        self.results_text.insert(tk.END, f"Number of features: {self.X_train.shape[1]}\n")
        
    def create_visualization_frame(self):
        """Create the frame for visualizations"""
        viz_frame = ttk.LabelFrame(self.root, text="Visualization")
        viz_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create a Figure and Canvas for plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def select_discharges(self):
        """Open a dialog to manually select discharges for training and testing"""
        # Get all discharge IDs
        all_discharge_ids = list(self.classifier.discharge_labels.keys())
        
        # Open the selection dialog
        dialog = DischargeSelectionDialog(self.root, all_discharge_ids, self.selected_discharges)
        self.root.wait_window(dialog)
        
        if dialog.result:
            self.selected_discharges = dialog.result
            # Update the data split
            self.update_data_split()
            # Update the dataset info display
            self.show_dataset_info()
            
            self.results_text.insert(tk.END, "\nDischarge selection updated:\n")
            self.results_text.insert(tk.END, f"- Training set: {len(self.selected_discharges['train'])} discharges\n")
            self.results_text.insert(tk.END, f"- Test set: {len(self.selected_discharges['test'])} discharges\n")
            self.results_text.see(tk.END)
            
    def update_data_split(self):
        """Update the training/test data based on selected discharges"""
        if not self.selected_discharges:
            return
            
        # Get the X, y data points for selected discharges
        X_scaled, y, discharge_ids = self.classifier.prepare_features()
        
        # Create masks for train and test sets
        train_mask = np.array([did in self.selected_discharges["train"] for did in discharge_ids])
        test_mask = np.array([did in self.selected_discharges["test"] for did in discharge_ids])
        
        # Split the data
        self.X_train = X_scaled[train_mask]
        self.y_train = y[train_mask]
        self.ids_train = np.array(discharge_ids)[train_mask]
        
        self.X_test = X_scaled[test_mask]
        self.y_test = y[test_mask]
        self.ids_test = np.array(discharge_ids)[test_mask]
        
        # Extract normal data for training (as in original code)
        self.X_train_normal = self.X_train[self.y_train == 0]
        self.X_train_disruptive = self.X_train[self.y_train == 1]
        
        print(f"Data split updated. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        print(f"Normal samples for training: {self.X_train_normal.shape[0]}")
        print(f"Disruptive samples for training: {self.X_train_disruptive.shape[0]}")
    
    def open_autotune_dialog(self):
        """Open a dialog to configure autotune parameters"""
        dialog = AutotuneConfigDialog(self.root)
        self.root.wait_window(dialog)
        
        if dialog.result:
            self.autotune_params = dialog.result
            self.autotune()
            
    def train_and_test(self):
        """Train IForest with current parameters and evaluate"""
        # Get parameters
        n_estimators = self.n_estimators_var.get()
        contamination = self.contamination_var.get()
        max_features = self.max_features_var.get()
        max_samples = self.max_samples_var.get()
        normal_only = self.normal_only_var.get()
        
        # Print current parameters
        self.results_text.insert(tk.END, f"\n\n=== New Training Run ===\n")
        self.results_text.insert(tk.END, f"Parameters: \n")
        self.results_text.insert(tk.END, f"- n_estimators: {n_estimators}\n")
        self.results_text.insert(tk.END, f"- contamination: {contamination:.3f}\n")
        self.results_text.insert(tk.END, f"- max_features: {max_features:.3f}\n")
        self.results_text.insert(tk.END, f"- max_samples: {max_samples:.3f}\n")
        self.results_text.insert(tk.END, f"- normal_only: {normal_only}\n")
        self.results_text.see(tk.END)
        
        # Create and train the model
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            max_samples=max_samples,
            random_state=42
        )
        
        if normal_only:
            X_train_data = self.X_train_normal
            self.results_text.insert(tk.END, f"Training on {len(X_train_data)} normal samples only\n")
            self.results_text.insert(tk.END, f"This is the standard approach for anomaly detection\n")
        else:
            X_train_data = self.X_train
            self.results_text.insert(tk.END, f"Training on all {len(X_train_data)} samples\n")
            self.results_text.insert(tk.END, f"  - Normal samples: {np.sum(self.y_train == 0)}\n")
            self.results_text.insert(tk.END, f"  - Disruptive samples: {np.sum(self.y_train == 1)}\n")
            
        model.fit(X_train_data)
        
        # Get detailed information about anomaly scores
        train_scores = -model.score_samples(self.X_train)
        test_scores = -model.score_samples(self.X_test)
        
        # Calculate threshold from training scores
        threshold = np.percentile(train_scores, 100 * (1 - contamination))
        self.results_text.insert(tk.END, f"\nAnomaly score threshold: {threshold:.4f}\n")
        
        # Debug score distributions
        self.results_text.insert(tk.END, f"\n=== Score Distributions ===\n")
        
        # Training set scores
        train_normal_scores = train_scores[self.y_train == 0]
        train_disruptive_scores = train_scores[self.y_train == 1]
        
        self.results_text.insert(tk.END, f"Training scores (normal): min={np.min(train_normal_scores):.4f}, "
                           f"mean={np.mean(train_normal_scores):.4f}, max={np.max(train_normal_scores):.4f}\n")
        if len(train_disruptive_scores) > 0:
            self.results_text.insert(tk.END, f"Training scores (disruptive): min={np.min(train_disruptive_scores):.4f}, "
                               f"mean={np.mean(train_disruptive_scores)}, max={np.max(train_disruptive_scores):.4f}\n")
        
        # Test set scores
        test_normal_scores = test_scores[self.y_test == 0]
        test_disruptive_scores = test_scores[self.y_test == 1]
        
        self.results_text.insert(tk.END, f"Test scores (normal): min={np.min(test_normal_scores):.4f}, "
                           f"mean={np.mean(test_normal_scores):.4f}, max={np.max(test_normal_scores):.4f}\n")
        self.results_text.insert(tk.END, f"Test scores (disruptive): min={np.min(test_disruptive_scores):.4f}, "
                           f"mean={np.mean(test_disruptive_scores):.4f}, max={np.max(test_disruptive_scores):.4f}\n")
        
        # How many samples are classified as anomalies based on the threshold
        train_anomaly_count = np.sum(train_scores > threshold)
        test_anomaly_count = np.sum(test_scores > threshold)
        
        self.results_text.insert(tk.END, f"\nSamples above threshold (predicted anomalies):\n")
        self.results_text.insert(tk.END, f"- Training set: {train_anomaly_count} / {len(train_scores)} "
                           f"({train_anomaly_count/len(train_scores)*100:.2f}%)\n")
        self.results_text.insert(tk.END, f"- Test set: {test_anomaly_count} / {len(test_scores)} "
                           f"({test_anomaly_count/len(test_scores)*100:.2f}%)\n")
        
        # Evaluate on training data
        y_train_pred_raw = model.predict(self.X_train)
        y_train_pred = (y_train_pred_raw == -1).astype(int)  # Convert to binary (1 = anomaly)
        
        # Evaluate on test data
        y_test_pred_raw = model.predict(self.X_test)
        y_test_pred = (y_test_pred_raw == -1).astype(int)  # Convert to binary (1 = anomaly)
        
        # Add debugging info about predictions
        self.results_text.insert(tk.END, f"\n=== Prediction Counts ===\n")
        self.results_text.insert(tk.END, f"Training predictions: {np.sum(y_train_pred == 0)} normal, {np.sum(y_train_pred == 1)} disruptive\n")
        self.results_text.insert(tk.END, f"Test predictions: {np.sum(y_test_pred == 0)} normal, {np.sum(y_test_pred == 1)} disruptive\n")
        
        if np.sum(y_test_pred == 1) == 0:
            self.results_text.insert(tk.END, "\n⚠️ WARNING: Model is predicting ALL test samples as non-disruptive!\n")
            self.results_text.insert(tk.END, "This will cause F1 score to be 0. Try adjusting contamination parameter.\n\n")
        
        # Calculate metrics with zero_division parameter
        train_acc = accuracy_score(self.y_train, y_train_pred)
        train_prec = precision_score(self.y_train, y_train_pred, zero_division=0)
        train_rec = recall_score(self.y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, zero_division=0)
        
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_prec = precision_score(self.y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(self.y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
        
        # Store detailed prediction information
        disruptive_test_indices = np.where(self.y_test == 1)[0]
        disruptive_predictions = y_test_pred[disruptive_test_indices]
        disruptive_discharge_ids = [self.ids_test[i] for i in disruptive_test_indices]
        
        self.results_text.insert(tk.END, f"\n=== Predictions on Disruptive Test Samples ===\n")
        for i, (idx, pred, did) in enumerate(zip(disruptive_test_indices, disruptive_predictions, disruptive_discharge_ids)):
            self.results_text.insert(tk.END, f"Discharge {did}: {'Correctly predicted' if pred == 1 else 'Incorrectly predicted'}, "
                               f"Score: {test_scores[idx]:.4f} (Threshold: {threshold:.4f})\n")
        
        # Store results
        self.train_results = {
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1,
            'predictions': y_train_pred,
            'true': self.y_train,
            'scores': train_scores
        }
        
        self.test_results = {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1,
            'predictions': y_test_pred,
            'true': self.y_test,
            'scores': test_scores
        }
        
        # Print results
        self.results_text.insert(tk.END, "\nTraining Results:\n")
        self.results_text.insert(tk.END, f"- Accuracy: {train_acc:.4f}\n")
        self.results_text.insert(tk.END, f"- Precision: {train_prec:.4f}\n")
        self.results_text.insert(tk.END, f"- Recall: {train_rec:.4f}\n")
        self.results_text.insert(tk.END, f"- F1 Score: {train_f1:.4f}\n")
        
        self.results_text.insert(tk.END, "\nTest Results:\n")
        self.results_text.insert(tk.END, f"- Accuracy: {test_acc:.4f}\n")
        self.results_text.insert(tk.END, f"- Precision: {test_prec:.4f}\n")
        self.results_text.insert(tk.END, f"- Recall: {test_rec:.4f}\n")
        self.results_text.insert(tk.END, f"- F1 Score: {test_f1:.4f}\n")
        self.results_text.see(tk.END)
        
        # Check if this is the best model so far
        if test_f1 > self.best_score:
            self.best_score = test_f1
            self.best_params = {
                'n_estimators': n_estimators,
                'contamination': contamination,
                'max_features': max_features,
                'max_samples': max_samples
            }
            self.results_text.insert(tk.END, "\n*** New Best Model! ***\n")
            self.results_text.see(tk.END)
            
        # Create confusion matrices and visualize
        self.visualize_results(y_train_pred, y_test_pred, train_scores, test_scores)
        
    def train_with_params(self, n_estimators, contamination, max_features, max_samples, normal_only=True, verbose=True):
        """Train and evaluate with specific parameters, returning the test F1 score"""
        # Create and train the model
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            max_samples=max_samples,
            random_state=42
        )
        
        if normal_only:
            X_train_data = self.X_train_normal
        else:
            X_train_data = self.X_train
            
        model.fit(X_train_data)
        
        # Evaluate on test data
        y_test_pred_raw = model.predict(self.X_test)
        y_test_pred = (y_test_pred_raw == -1).astype(int)  # Convert to binary (1 = anomaly)
        test_scores = -model.score_samples(self.X_test)
        
        # Calculate metrics
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_prec = precision_score(self.y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(self.y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
        
        # Calculate alternative metrics that work even when F1 is 0
        # Anomaly detection specific: how well anomaly scores separate classes
        disruptive_idx = self.y_test == 1
        normal_idx = self.y_test == 0
        
        # Use absolute difference to ensure positive values
        score_separation = abs(np.mean(test_scores[disruptive_idx]) - np.mean(test_scores[normal_idx]))
        
        if verbose:
            return test_f1, test_acc, test_prec, test_rec, score_separation
        
        # At the end of train_with_params, before returning
        self.results_text.insert(tk.END, f"Debug - predictions: {np.sum(y_test_pred == 0)} normal, {np.sum(y_test_pred == 1)} disruptive\n")

        # Return both F1 and score separation to use as backup
        return test_f1, score_separation
        
    def autotune(self):
        """Automatically search for best parameter combination using custom parameter ranges"""
        # Use the configured parameter ranges
        if not hasattr(self, 'autotune_params'):
            # If not configured, use defaults with wider contamination range
            n_estimators_grid = [50, 100, 200, 300]
            contamination_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            max_features_grid = [0.5, 0.7, 0.9, 1.0]
            max_samples_grid = [0.5, 0.7, 0.9, 1.0]
        else:
            n_estimators_grid = self.autotune_params["n_estimators"]
            contamination_grid = self.autotune_params["contamination"]
            max_features_grid = self.autotune_params["max_features"]
            max_samples_grid = self.autotune_params["max_samples"]
        
        # Define whether to train on normal data only or all data
        normal_only = self.normal_only_var.get()
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=== Starting Autotune Process ===\n")
        self.results_text.insert(tk.END, "This may take some time...\n\n")
        self.results_text.insert(tk.END, f"Training on {'normal samples only' if normal_only else 'all samples'}\n\n")
        self.results_text.insert(tk.END, "Parameter grid:\n")
        self.results_text.insert(tk.END, f"- n_estimators: {n_estimators_grid}\n")
        self.results_text.insert(tk.END, f"- contamination: {contamination_grid}\n")
        self.results_text.insert(tk.END, f"- max_features: {max_features_grid}\n")
        self.results_text.insert(tk.END, f"- max_samples: {max_samples_grid}\n\n")
        
        total_combinations = len(n_estimators_grid) * len(contamination_grid) * len(max_features_grid) * len(max_samples_grid)
        self.results_text.insert(tk.END, f"Total configurations to test: {total_combinations}\n\n")
        self.results_text.see(tk.END)
        self.root.update()
        
        # Track progress and best parameters
        start_time = time.time()
        count = 0
        best_f1 = -1
        best_separation = -float('inf')  # For backup metric
        best_params = None
        best_params_by_separation = None
        results_list = []
        
        # Create progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            self.root, 
            variable=progress_var,
            maximum=total_combinations
        )
        progress_bar.grid(row=2, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        progress_label = ttk.Label(self.root, text="Progress: 0%")
        progress_label.grid(row=2, column=0, columnspan=2, sticky="e", padx=15, pady=5)
        
        try:
            # Try all combinations
            for n_est, cont, max_feat, max_samp in itertools.product(
                n_estimators_grid, contamination_grid, max_features_grid, max_samples_grid
            ):
                # Update progress
                count += 1
                progress_var.set(count)
                progress_label.config(text=f"Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")
                self.root.update()
                
                # Train and evaluate with these parameters
                f1, score_separation = self.train_with_params(n_est, cont, max_feat, max_samp, normal_only, verbose=False)
                
                # Store result with both metrics
                result = {
                    'n_estimators': n_est,
                    'contamination': cont,
                    'max_features': max_feat,
                    'max_samples': max_samp,
                    'f1_score': f1,
                    'score_separation': score_separation
                }
                results_list.append(result)
                
                # Log all configurations performance to the debug window
                self.results_text.insert(tk.END, 
                    f"Config {count}/{total_combinations}: n_est={n_est}, cont={cont:.2f}, "
                    f"max_feat={max_feat:.2f}, max_samp={max_samp:.2f} → F1={f1:.4f}, Sep={score_separation:.4f}\n"
                )
                self.results_text.see(tk.END)
                
                # Check if this is best so far (F1 metric)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'n_estimators': n_est,
                        'contamination': cont,
                        'max_features': max_feat,
                        'max_samples': max_samp
                    }
                    
                    self.results_text.insert(tk.END, f"★ NEW BEST F1! F1={best_f1:.4f} with {best_params}\n")
                    self.results_text.see(tk.END)
                
                # Also track best separation as a backup metric
                if score_separation > best_separation:
                    best_separation = score_separation
                    best_params_by_separation = {
                        'n_estimators': n_est,
                        'contamination': cont,
                        'max_features': max_feat,
                        'max_samples': max_samp
                    }
                    
                    self.results_text.insert(tk.END, f"◆ New best separation: {best_separation:.4f} with {best_params_by_separation}\n")
                    self.results_text.see(tk.END)
                
                # Occasionally update display (every 5% of progress)
                if count % max(1, total_combinations // 20) == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / count) * (total_combinations - count)
                    self.results_text.insert(tk.END, f"Progress: {count}/{total_combinations}, ETA: {eta:.1f}s\n")
                    self.results_text.see(tk.END)
        
        finally:
            # Remove progress bar
            progress_bar.grid_forget()
            progress_label.grid_forget()
        
        # Show final results
        elapsed = time.time() - start_time
        self.results_text.insert(tk.END, f"\n=== Autotune Complete ===\n")
        self.results_text.insert(tk.END, f"Total time: {elapsed:.2f} seconds\n")
        
        # If F1 is always 0, use the backup metric
        if best_f1 == 0 and best_separation > -float('inf'):
            self.results_text.insert(tk.END, "\n⚠️ All configurations had F1=0. Using anomaly score separation metric instead.\n")
            best_params = best_params_by_separation
            
        if best_params:
            self.best_params = best_params
            self.best_score = best_f1
            
            self.results_text.insert(tk.END, f"\nBest Parameters Found:\n")
            self.results_text.insert(tk.END, f"- n_estimators: {best_params['n_estimators']}\n")
            self.results_text.insert(tk.END, f"- contamination: {best_params['contamination']}\n")
            self.results_text.insert(tk.END, f"- max_features: {best_params['max_features']}\n")
            self.results_text.insert(tk.END, f"- max_samples: {best_params['max_samples']}\n")
            
            if best_params == best_params_by_separation and best_f1 == 0:
                self.results_text.insert(tk.END, f"- Score Separation: {best_separation:.4f}\n")
            else:
                self.results_text.insert(tk.END, f"- F1 Score: {best_f1:.4f}\n")
            
            # Update the slider values to the best parameters
            self.n_estimators_var.set(best_params['n_estimators'])
            self.contamination_var.set(best_params['contamination'])
            self.max_features_var.set(best_params['max_features'])
            self.max_samples_var.set(best_params['max_samples'])
            
            # Show top 5 configurations - sort by F1 first, then by separation
            self.results_text.insert(tk.END, "\nTop 5 Configurations:\n")
            top_results = sorted(results_list, key=lambda x: (x['f1_score'], x['score_separation']), reverse=True)[:5]
            for i, result in enumerate(top_results, 1):
                self.results_text.insert(tk.END, f"{i}. F1={result['f1_score']:.4f}, Sep={result['score_separation']:.4f}: n_est={result['n_estimators']}, "
                                   f"cont={result['contamination']}, max_feat={result['max_features']}, "
                                   f"max_samp={result['max_samples']}\n")
            
            # Run the best configuration to show full results
            self.results_text.insert(tk.END, "\nRunning final evaluation with best parameters...\n")
            self.train_and_test()
        else:
            self.results_text.insert(tk.END, "No suitable parameters found.\n")
        
        self.results_text.see(tk.END)
        
    def visualize_results(self, y_train_pred, y_test_pred, train_scores=None, test_scores=None):
        """Visualize confusion matrices and other metrics"""
        self.fig.clear()
        
        if train_scores is not None and test_scores is not None:
            # Create 2x2 grid for more visualizations
            self.ax1 = self.fig.add_subplot(2, 2, 1)
            self.ax2 = self.fig.add_subplot(2, 2, 2)
            
            # Add score distribution plots
            ax3 = self.fig.add_subplot(2, 2, 3)
            ax4 = self.fig.add_subplot(2, 2, 4)
            
            # Plot score distributions for training
            ax3.hist(train_scores[self.y_train == 0], bins=20, alpha=0.5, label='Normal')
            if np.sum(self.y_train == 1) > 0:
                ax3.hist(train_scores[self.y_train == 1], bins=20, alpha=0.5, label='Disruptive')
            ax3.set_title("Training Anomaly Scores")
            ax3.set_xlabel("Anomaly Score")
            ax3.set_ylabel("Count")
            ax3.legend()
            
            # Plot score distributions for testing
            ax4.hist(test_scores[self.y_test == 0], bins=20, alpha=0.5, label='Normal')
            ax4.hist(test_scores[self.y_test == 1], bins=20, alpha=0.5, label='Disruptive')
            ax4.set_title("Test Anomaly Scores")
            ax4.set_xlabel("Anomaly Score")
            ax4.set_ylabel("Count")
            ax4.legend()
        else:
            # Use original 1x2 layout
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax2 = self.fig.add_subplot(1, 2, 2)
        
        # Training confusion matrix
        cm_train = confusion_matrix(self.y_train, y_train_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=self.ax1)
        self.ax1.set_title("Training Confusion Matrix")
        self.ax1.set_xlabel("Predicted")
        self.ax1.set_ylabel("Actual")
        
        # Test confusion matrix
        cm_test = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=self.ax2)
        self.ax2.set_title("Test Confusion Matrix")
        self.ax2.set_xlabel("Predicted")
        self.ax2.set_ylabel("Actual")
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
    def reset_best(self):
        """Reset the best model tracking"""
        self.best_params = None
        self.best_score = -1
        self.results_text.insert(tk.END, "\nBest model tracking reset.\n")
        self.results_text.see(tk.END)
        
    def save_best_config(self):
        """Save the best configuration to a JSON file"""
        if self.best_params is None:
            messagebox.showwarning("Warning", "No best parameters found yet. Train a model first.")
            return
            
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="c:\\Users\\Ruben\\Desktop"
        )
        
        if not file_path:
            return
            
        # Try to load existing config if file exists
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
        except:
            config = {}
            
        # Update only the iforest part of the configuration
        config['iforest'] = {
            'n_estimators': int(self.best_params['n_estimators']),
            'contamination': float(self.best_params['contamination']),
            'max_features': float(self.best_params['max_features']),
            'max_samples': float(self.best_params['max_samples'])
        }
        
        # Save the file
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.results_text.insert(tk.END, f"\nBest configuration saved to: {file_path}\n")
        self.results_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = IForestTunerApp(root)
    root.mainloop()
