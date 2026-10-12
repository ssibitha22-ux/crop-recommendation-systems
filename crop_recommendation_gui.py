import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Crop_recommendation.csv")

# Separate input features and target label
X = data.drop("label", axis=1)
y = data["label"]


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 3. TRAIN DECISION TREE MODEL
# ==========================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


# ==========================================
# 4. CALCULATE MODEL ACCURACY
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


# Store the latest recommendation
latest_result = None


# ==========================================
# 5. CROP INFORMATION
# ==========================================

crop_information = {
    "apple": (
        "Apple is a fruit crop that generally grows well "
        "in suitable cool climates and well-drained soil."
    ),

    "banana": (
        "Banana is a tropical fruit crop that generally "
        "needs warm conditions and sufficient water."
    ),

    "blackgram": (
        "Blackgram is a pulse crop used as a source of protein."
    ),

    "chickpea": (
        "Chickpea is a pulse crop commonly grown in "
        "relatively dry conditions."
    ),

    "coconut": (
        "Coconut is a tropical crop that grows well in "
        "warm conditions with suitable rainfall."
    ),

    "coffee": (
        "Coffee is a plantation crop that commonly grows "
        "in suitable tropical highland conditions."
    ),

    "cotton": (
        "Cotton is a fibre crop used in the textile industry."
    ),

    "grapes": (
        "Grapes are fruit crops that need suitable "
        "temperatures and well-drained soil."
    ),

    "jute": (
        "Jute is a fibre crop used to produce natural fibres."
    ),

    "kidneybeans": (
        "Kidney beans are pulse crops that provide protein."
    ),

    "lentil": (
        "Lentil is a pulse crop and a source of plant protein."
    ),

    "maize": (
        "Maize is a cereal crop used for food, feed and industry."
    ),

    "mango": (
        "Mango is a tropical fruit crop that grows on trees."
    ),

    "mothbeans": (
        "Mothbeans are drought-tolerant pulse crops."
    ),

    "mungbean": (
        "Mungbean is a pulse crop commonly known as green gram."
    ),

    "muskmelon": (
        "Muskmelon is a fruit crop that generally grows "
        "in warm conditions."
    ),

    "orange": (
        "Orange is a citrus fruit crop rich in vitamin C."
    ),

    "papaya": (
        "Papaya is a tropical fruit crop that grows in warm conditions."
    ),

    "pigeonpeas": (
        "Pigeon pea is a pulse crop that can tolerate "
        "relatively dry conditions."
    ),

    "pomegranate": (
        "Pomegranate is a fruit crop that can grow in "
        "suitable warm and relatively dry climates."
    ),

    "rice": (
        "Rice is a cereal crop commonly grown in warm "
        "conditions with suitable water availability."
    ),

    "watermelon": (
        "Watermelon is a fruit crop that generally grows "
        "well in warm weather."
    )
}


# ==========================================
# 6. CREATE MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Crop Recommendation System")
root.geometry("720x800")
root.configure(bg="#EAF7EA")
root.resizable(False, False)


# ==========================================
# 7. TITLE AND SUBTITLE
# ==========================================

title = tk.Label(
    root,
    text="CROP RECOMMENDATION SYSTEM",
    font=("Arial", 20, "bold"),
    bg="#EAF7EA",
    fg="#1B5E20"
)

title.pack(pady=15)


subtitle = tk.Label(
    root,
    text="Enter Soil and Weather Details",
    font=("Arial", 12),
    bg="#EAF7EA"
)

subtitle.pack(pady=5)


# ==========================================
# 8. INPUT FIELDS
# ==========================================

input_frame = tk.Frame(root, bg="#EAF7EA")
input_frame.pack(pady=8)


fields = [
    ("Nitrogen (N):", "90"),
    ("Phosphorus (P):", "42"),
    ("Potassium (K):", "43"),
    ("Temperature (°C):", "20.8"),
    ("Humidity (%):", "82"),
    ("Soil pH:", "6.5"),
    ("Rainfall (mm):", "203")
]


entries = {}


for row, (label_text, default_value) in enumerate(fields):

    label = tk.Label(
        input_frame,
        text=label_text,
        font=("Arial", 11),
        bg="#EAF7EA",
        width=20,
        anchor="w"
    )

    label.grid(
        row=row,
        column=0,
        padx=10,
        pady=5
    )

    entry = tk.Entry(
        input_frame,
        font=("Arial", 11),
        width=22
    )

    entry.grid(
        row=row,
        column=1,
        padx=10,
        pady=5
    )

    entry.insert(0, default_value)

    entries[label_text] = entry


# ==========================================
# 9. RECOMMEND A CROP
# ==========================================

def recommend_crop():

    global latest_result

    try:

        # Read input values
        N = float(entries["Nitrogen (N):"].get())
        P = float(entries["Phosphorus (P):"].get())
        K = float(entries["Potassium (K):"].get())

        temperature = float(
            entries["Temperature (°C):"].get()
        )

        humidity = float(
            entries["Humidity (%):"].get()
        )

        ph = float(
            entries["Soil pH:"].get()
        )

        rainfall = float(
            entries["Rainfall (mm):"].get()
        )

        # Use NumPy to check numeric values
        values = np.array(
            [N, P, K, temperature, humidity, ph, rainfall],
            dtype=float
        )

        # Reject infinity and NaN values
        if not np.isfinite(values).all():

            messagebox.showerror(
                "Input Error",
                "Please enter valid finite numeric values."
            )

            return

        # Check nutrient and rainfall values
        if N < 0 or P < 0 or K < 0 or rainfall < 0:

            messagebox.showerror(
                "Input Error",
                "N, P, K and rainfall cannot be negative."
            )

            return

        # Check temperature
        if not -10 <= temperature <= 60:

            messagebox.showerror(
                "Input Error",
                "Temperature must be between -10 and 60 degrees C."
            )

            return

        # Check humidity
        if not 0 <= humidity <= 100:

            messagebox.showerror(
                "Input Error",
                "Humidity must be between 0 and 100."
            )

            return

        # Check soil pH
        if not 0 <= ph <= 14:

            messagebox.showerror(
                "Input Error",
                "Soil pH must be between 0 and 14."
            )

            return

        # Warn about unusual soil pH
        if ph < 3 or ph > 10:

            messagebox.showwarning(
                "Unusual Soil pH",
                "Please verify this soil pH value before continuing."
            )

        # Prepare input for the machine learning model
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=X.columns
        )

        # Predict the recommended crop
        crop = model.predict(input_data)[0]

        # Store the latest result
        latest_result = {
            "Nitrogen": N,
            "Phosphorus": P,
            "Potassium": K,
            "Temperature": temperature,
            "Humidity": humidity,
            "Soil_pH": ph,
            "Rainfall": rainfall,
            "Recommended_Crop": crop
        }

        # Display recommendation
        result_label.config(
            text="Recommended Crop: " + crop.capitalize(),
            fg="#1B5E20"
        )

        # Display crop information
        info_label.config(
            text=crop_information.get(
                crop.lower(),
                "No additional information is available."
            )
        )

    except ValueError:

        messagebox.showerror(
            "Input Error",
            "Please enter valid numbers in all input boxes."
        )


# ==========================================
# 10. SHOW MODEL ACCURACY
# ==========================================

def show_accuracy():

    messagebox.showinfo(
        "Model Performance",
        f"Algorithm: Decision Tree Classifier\n"
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
        "Accuracy is measured on the test dataset."
    )


# ==========================================
# 11. SHOW AVAILABLE CROPS
# ==========================================

def show_crops():

    crops = sorted(data["label"].unique())

    crop_list = "\n".join(
        f"{number}. {crop.capitalize()}"
        for number, crop in enumerate(crops, start=1)
    )

    messagebox.showinfo(
        "Available Crops",
        f"Total crops: {len(crops)}\n\n{crop_list}"
    )


# ==========================================
# 12. SAVE RECOMMENDATION
# ==========================================

def save_result():

    if latest_result is None:

        messagebox.showwarning(
            "No Result",
            "Please recommend a crop before saving the result."
        )

        return

    filename = "crop_recommendation_results.csv"

    try:

        new_data = pd.DataFrame([latest_result])

        # Append if the file already exists
        if os.path.exists(filename):

            new_data.to_csv(
                filename,
                mode="a",
                header=False,
                index=False
            )

        # Create a new file if it does not exist
        else:

            new_data.to_csv(
                filename,
                index=False
            )

        messagebox.showinfo(
            "Saved Successfully",
            "Your recommendation has been saved to:\n"
            + os.path.abspath(filename)
        )

    except OSError as error:

        messagebox.showerror(
            "Save Error",
            f"Could not save the result.\n{error}"
        )


# ==========================================
# 13. CLEAR INPUT FIELDS
# ==========================================

def clear_fields():

    global latest_result

    # Clear every input box
    for entry in entries.values():

        entry.delete(0, tk.END)

    # Reset the latest result
    latest_result = None

    # Reset recommendation display
    result_label.config(
        text="Recommended Crop: --",
        fg="#333333"
    )

    # Clear crop information
    info_label.config(text="")


# ==========================================
# 14. CREATE BUTTONS
# ==========================================

button_frame = tk.Frame(root, bg="#EAF7EA")
button_frame.pack(pady=10)


# Recommend Crop button
tk.Button(
    button_frame,
    text="Recommend Crop",
    font=("Arial", 10, "bold"),
    bg="#2E7D32",
    fg="white",
    width=17,
    command=recommend_crop
).grid(
    row=0,
    column=0,
    padx=5,
    pady=5
)


# View Accuracy button
tk.Button(
    button_frame,
    text="View Accuracy",
    font=("Arial", 10, "bold"),
    bg="#1565C0",
    fg="white",
    width=17,
    command=show_accuracy
).grid(
    row=0,
    column=1,
    padx=5,
    pady=5
)


# View Available Crops button
tk.Button(
    button_frame,
    text="View Available Crops",
    font=("Arial", 10, "bold"),
    bg="#EF6C00",
    fg="white",
    width=17,
    command=show_crops
).grid(
    row=1,
    column=0,
    padx=5,
    pady=5
)


# Save Result button
tk.Button(
    button_frame,
    text="Save Result",
    font=("Arial", 10, "bold"),
    bg="#00838F",
    fg="white",
    width=17,
    command=save_result
).grid(
    row=1,
    column=1,
    padx=5,
    pady=5
)


# Clear Fields button
tk.Button(
    button_frame,
    text="Clear Fields",
    font=("Arial", 10, "bold"),
    width=17,
    command=clear_fields
).grid(
    row=2,
    column=0,
    columnspan=2,
    pady=5
)


# ==========================================
# 15. DISPLAY RECOMMENDATION
# ==========================================

result_label = tk.Label(
    root,
    text="Recommended Crop: --",
    font=("Arial", 15, "bold"),
    bg="#EAF7EA",
    fg="#333333"
)

result_label.pack(pady=8)


# ==========================================
# 16. DISPLAY CROP INFORMATION
# ==========================================

info_label = tk.Label(
    root,
    text="",
    font=("Arial", 10),
    bg="#EAF7EA",
    fg="#333333",
    wraplength=620,
    justify="center"
)

info_label.pack(pady=5)


# ==========================================
# 17. FOOTER
# ==========================================

footer = tk.Label(
    root,
    text="Machine Learning | Decision Tree Classifier",
    font=("Arial", 9),
    bg="#EAF7EA",
    fg="#555555"
)

footer.pack(
    side="bottom",
    pady=10
)


# ==========================================
# 18. RUN APPLICATION
# ==========================================

root.mainloop()
