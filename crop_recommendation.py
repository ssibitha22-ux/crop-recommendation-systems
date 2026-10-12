import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Separate input features and output
X = data.drop("label", axis=1)
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)


# Function to recommend a crop
def recommend_crop():
    print("\n===== ENTER SOIL AND WEATHER DETAILS =====")

    try:
        N = float(input("Enter Nitrogen level: "))
        P = float(input("Enter Phosphorus level: "))
        K = float(input("Enter Potassium level: "))
        temperature = float(input("Enter Temperature (C): "))
        humidity = float(input("Enter Humidity (%): "))
        ph = float(input("Enter Soil pH: "))
        rainfall = float(input("Enter Rainfall (mm): "))

        # Validate inputs
        if N < 0 or P < 0 or K < 0 or rainfall < 0:
            print("Error: Values cannot be negative.")
            return

        if not 0 <= humidity <= 100:
            print("Error: Humidity must be between 0 and 100.")
            return

        if not 0 <= ph <= 14:
            print("Error: Soil pH must be between 0 and 14.")
            return

        # Prepare data for prediction
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=X.columns
        )

        # Predict the crop
        crop = model.predict(input_data)[0]

        print("\n====================================")
        print("       CROP RECOMMENDATION")
        print("====================================")
        print("Recommended Crop:", crop.capitalize())
        print("====================================")

    except ValueError:
        print("Error: Please enter valid numeric values.")


# Function to display model accuracy
def show_accuracy():
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Algorithm: Decision Tree Classifier")
    print("====================================")


# Function to display available crops
def show_crops():
    print("\n===== AVAILABLE CROPS =====")

    crops = sorted(data["label"].unique())

    for number, crop in enumerate(crops, start=1):
        print(f"{number}. {crop.capitalize()}")

    print("\nTotal number of crops:", len(crops))


# Main menu
def main():
    while True:
        print("\n====================================")
        print("      CROP RECOMMENDATION ENGINE")
        print("====================================")
        print("1. Recommend a Crop")
        print("2. View Model Accuracy")
        print("3. View Available Crops")
        print("4. Exit")
        print("====================================")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            recommend_crop()

        elif choice == "2":
            show_accuracy()

        elif choice == "3":
            show_crops()

        elif choice == "4":
            print("\nThank you for using Crop Recommendation Engine!")
            break

        else:
            print("\nInvalid choice. Please enter a number from 1 to 4.")


# Run the program
if __name__ == "__main__":
    main()
