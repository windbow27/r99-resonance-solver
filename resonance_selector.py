import tkinter as tk
from tkinter import ttk
import threading
from resonance_solver import main, AttackBuild, DefBuild, CritBuild

def run_main(resonanceLevel, resonanceType, criteria, result_text, status_label):
    def target():
        status_label.config(text=f"Level: {resonanceLevel}, Type: {resonanceType}, {criteria.__class__.__name__}\nRunning... Please wait...")

        result = main(resonanceLevel, resonanceType, criteria)

        # Update the Text widget with the result
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        if result is not None:
            result_text.insert(tk.END, format_result(result))
        else:
            result_text.insert(tk.END, "No result returned from main function.")
        result_text.config(state=tk.DISABLED)

        status_label.config(text="")
    
    threading.Thread(target=target).start()

def format_result(result):
    formatted_result = ""
    for item in result:
        formatted_result += f"{item}\n"
    return formatted_result

def create_ui():
    root = tk.Tk()
    root.title("Resonance Solver")

    # Dropdowns for resonance level, resonance type, and criteria
    tk.Label(root, text="Resonance Level:").grid(row=0, column=0, padx=10, pady=10)
    resonance_level_var = tk.IntVar(value=10)
    resonance_level_menu = ttk.Combobox(root, textvariable=resonance_level_var, values=list(range(2, 16)))
    resonance_level_menu.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Resonance Type:").grid(row=1, column=0, padx=10, pady=10)
    resonance_type_var = tk.StringVar(value='Z')
    resonance_type_menu = ttk.Combobox(root, textvariable=resonance_type_var, values=['Z', 'T', 'U', 'Plus'])
    resonance_type_menu.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Criteria:").grid(row=2, column=0, padx=10, pady=10)
    criteria_var = tk.StringVar(value='AttackBuild')
    criteria_menu = ttk.Combobox(root, textvariable=criteria_var, values=['AttackBuild', 'DefBuild', 'CritBuild'])
    criteria_menu.grid(row=2, column=1, padx=10, pady=10)

    # Run button
    def on_run_button_click():
        resonanceLevel = resonance_level_var.get()
        resonanceType = resonance_type_var.get()
        criteria_str = criteria_var.get()
        criteria = {'AttackBuild': AttackBuild(), 'DefBuild': DefBuild(), 'CritBuild': CritBuild()}[criteria_str]
        run_main(resonanceLevel, resonanceType, criteria, result_text, status_label)

    run_button = tk.Button(root, text="Run", command=on_run_button_click)
    run_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Text widget
    result_text = tk.Text(root, height=20, width=80, state=tk.DISABLED)
    result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    # Status label
    status_label = tk.Label(root, text="", fg="blue")
    status_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_ui()