import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt


def load_data():
    file_path = filedialog.askopenfilename()
    if file_path:
        weather = pd.read_csv(file_path, index_col='DATE')
        core_weather = weather[["PRCP", "TMAX", "TMIN"]].copy()
        core_weather["PRCP"] = core_weather["PRCP"].fillna(0.01)
        core_weather["TMAX"] = core_weather["TMAX"].ffill()
        core_weather["TMIN"] = core_weather["TMIN"].ffill()
        try:
            core_weather.index = pd.to_datetime(core_weather.index, format="%Y-%m-%d")
        except ValueError as e:
            messagebox.showerror("Date Parsing Error", f"Could not parse dates: {e}")
            return None
        core_weather["Target_TMAX"] = core_weather.shift(-1)["TMAX"]
        core_weather["Target_TMIN"] = core_weather.shift(-1)["TMIN"]
        core_weather = core_weather.iloc[:-1, :].copy()
        messagebox.showinfo("Data Loaded", "Weather data loaded successfully!")
        return core_weather
    else:
        messagebox.showwarning("No File Selected", "Please select a file to load data.")
        return None


def train_model(core_weather):
    predictors = ["PRCP", "TMAX", "TMIN"]
    reg_TMAX = Ridge(alpha=.1)
    reg_TMIN = Ridge(alpha=.1)
    train = core_weather.loc["2010-11-16":"2024-11-16"]
    test = core_weather.loc["2024-11-17":]

    reg_TMAX.fit(train[predictors], train["Target_TMAX"])
    reg_TMIN.fit(train[predictors], train["Target_TMIN"])

    messagebox.showinfo("Training Complete", "Model training is complete. You can now predict temperatures.")
    return reg_TMAX, reg_TMIN, test


def predict_temperature(reg_TMAX, reg_TMIN, core_weather):
    predictors = ["PRCP", "TMAX", "TMIN"]
    test = core_weather.loc["2024-11-17":]
    predictions_TMAX = reg_TMAX.predict(test[predictors])
    predictions_TMIN = reg_TMIN.predict(test[predictors])
    mae_TMAX = mean_absolute_error(test["Target_TMAX"], predictions_TMAX)
    mae_TMIN = mean_absolute_error(test["Target_TMIN"], predictions_TMIN)
    messagebox.showinfo("Prediction Complete",
                        f"Predictions are complete.\nMAE for TMAX: {mae_TMAX}\nMAE for TMIN: {mae_TMIN}")
    return test, predictions_TMAX, predictions_TMIN


def show_prediction_window(test, predictions_TMAX, predictions_TMIN):
    # Create a new window for displaying predictions
    prediction_window = tk.Toplevel()
    prediction_window.title("Predicted and Actual Temperatures")
    prediction_window.geometry("600x400")

    # Create a treeview to display the data
    columns = ("Date", "Actual TMAX", "Predicted TMAX", "Actual TMIN", "Predicted TMIN")
    tree = ttk.Treeview(prediction_window, columns=columns, show="headings")
    tree.pack(fill="both", expand=True)

    # Define the column headings
    for col in columns:
        tree.heading(col, text=col)

    # Populate the treeview with actual and predicted temperatures
    for date, actual_TMAX, actual_TMIN, pred_TMAX, pred_TMIN in zip(test.index, test["Target_TMAX"], test["Target_TMIN"], predictions_TMAX, predictions_TMIN):
        tree.insert("", "end", values=(date.strftime("%Y-%m-%d"), actual_TMAX, pred_TMAX, actual_TMIN, pred_TMIN))


def plot_predictions(test, predictions_TMAX, predictions_TMIN):
    # Plot TMAX predictions
    combined_TMAX = pd.concat([test["Target_TMAX"], pd.Series(predictions_TMAX, index=test.index)], axis=1)
    combined_TMAX.columns = ["Actual TMAX", "Predicted TMAX"]
    combined_TMAX.plot(figsize=(10, 6))
    plt.title("Actual vs Predicted TMAX")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    # Plot TMIN predictions
    combined_TMIN = pd.concat([test["Target_TMIN"], pd.Series(predictions_TMIN, index=test.index)], axis=1)
    combined_TMIN.columns = ["Actual TMIN", "Predicted TMIN"]
    combined_TMIN.plot(figsize=(10, 6))
    plt.title("Actual vs Predicted TMIN")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()


def display_data(tree, core_weather, predictions_TMAX=None, predictions_TMIN=None):
    for col in tree.get_children():
        tree.delete(col)
    for index, row in core_weather.iterrows():
        values = [index, row["PRCP"], row["TMAX"], row["TMIN"]]
        if predictions_TMAX is not None and predictions_TMIN is not None:
            pred_TMAX = predictions_TMAX.loc[index] if index in predictions_TMAX.index else ''
            pred_TMIN = predictions_TMIN.loc[index] if index in predictions_TMIN.index else ''
            values.extend([pred_TMAX, pred_TMIN])
        else:
            values.extend(['', ''])  # Placeholder if no predictions yet
        tree.insert("", "end", values=values)


def main():
    root = tk.Tk()
    root.title("Temperature Predictor")
    root.state('zoomed')
    root.configure(bg="#D1E7E0")  # Light blue-green background

    style = ttk.Style()
    style.configure("TButton",
                    font=("Helvetica", 16),  # Increased font size for better readability
                    padding=10,
                    background="#4CAF50",
                    foreground="black")
    style.map("TButton",
              background=[("active", "#45a049")])

    core_weather = None
    reg_TMAX = None
    reg_TMIN = None
    test = None
    predictions_TMAX = None
    predictions_TMIN = None

    def load_data_button():
        nonlocal core_weather
        core_weather = load_data()
        if core_weather is not None:
            display_data(tree, core_weather)

    def train_model_button():
        nonlocal reg_TMAX, reg_TMIN, test
        if core_weather is not None:
            reg_TMAX, reg_TMIN, test = train_model(core_weather)
        else:
            messagebox.showwarning("No Data", "Please load the data first.")

    def predict_temperature_button():
        nonlocal predictions_TMAX, predictions_TMIN
        if reg_TMAX is not None and reg_TMIN is not None:
            test, predictions_TMAX, predictions_TMIN = predict_temperature(reg_TMAX, reg_TMIN, core_weather)
            predictions_TMAX_series = pd.Series(predictions_TMAX, index=test.index)
            predictions_TMIN_series = pd.Series(predictions_TMIN, index=test.index)
            display_data(tree, core_weather, predictions_TMAX_series, predictions_TMIN_series)
            # After predictions are made, open the new window to show predicted and actual temperatures
            show_prediction_window(test, predictions_TMAX, predictions_TMIN)
        else:
            messagebox.showwarning("No Model", "Please train the model first.")

    def show_graph_button():
        if predictions_TMAX is not None and predictions_TMIN is not None:
            plot_predictions(test, predictions_TMAX, predictions_TMIN)
        else:
            messagebox.showwarning("No Predictions", "Please predict the temperatures first.")

    frame = tk.Frame(root, bg="#D1E7E0")
    frame.pack(pady=10)

    ttk.Button(frame, text="Load Data", command=load_data_button).grid(row=0, column=0, padx=10)
    ttk.Button(frame, text="Train Model", command=train_model_button).grid(row=0, column=1, padx=10)
    ttk.Button(frame, text="Predict Temperature", command=predict_temperature_button).grid(row=0, column=2, padx=10)
    ttk.Button(frame, text="Show Graph", command=show_graph_button).grid(row=0, column=3, padx=10)

    tree_frame = tk.Frame(root)
    tree_frame.pack(pady=20, fill="both", expand=True)

    tree_scroll = ttk.Scrollbar(tree_frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    columns = ("Date", "PRCP", "TMAX", "TMIN", "Predicted_TMAX", "Predicted_TMIN")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", yscrollcommand=tree_scroll.set)

    tree_scroll.config(command=tree.yview)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    tree.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
