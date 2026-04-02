# ✈️ Flight Fare Prediction Web Application

This project is a machine learning powered web application that predicts airline ticket prices based on trip details such as travel dates, source city, destination city, airline carrier, and number of stops. The goal of the project is to demonstrate how machine learning models can be integrated into an interactive web application that generates real-time predictions based on user input.

Flight price prediction is a classical forecasting problem where historical flight data is analyzed to identify patterns in airline pricing. Many modern travel platforms such as Google Flights provide insights like price trends and fare predictions to help travelers determine the best time to purchase tickets. This project recreates a simplified version of that concept by building a predictive model and deploying it through a user-friendly web interface.

The machine learning model was trained using a dataset containing historical flight information including airline carriers, travel routes, departure and arrival times, number of stops, and ticket prices. Feature engineering techniques were applied to transform the raw data into a format suitable for model training. For example, departure and arrival times were decomposed into hour and minute components, travel duration was calculated, and categorical variables such as airline, source, and destination were converted into numerical features using one-hot encoding.

The trained Random Forest regression model is stored as a serialized model file and integrated into a Streamlit web application. The interface allows users to enter travel details through dropdown menus, date selectors, and time inputs. After the user submits the form, the application processes the input data, converts it into the same feature format used during training, and generates a predicted flight ticket price instantly.

The application also includes a modern interface design featuring gradient backgrounds, glassmorphism-style cards, and a trip summary panel that dynamically displays selected travel information such as source, destination, number of stops, and flight duration.

This project demonstrates the complete machine learning workflow including data preprocessing, feature engineering, model training, and deployment through an interactive web interface.

---

# 🚀 Features

• Predict airline ticket prices based on travel details  
• Interactive web interface built with Streamlit  
• Real-time price prediction using a trained Random Forest model  
• Feature engineering for time and categorical variables  
• Trip summary dashboard displaying selected travel parameters  
• Modern responsive UI design

---

# 🧠 Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Streamlit

---

# 📊 Dataset

The dataset used for training the machine learning model is publicly available on Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh

The dataset contains historical flight information including:

- Airline carriers
- Source and destination cities
- Departure and arrival times
- Total number of stops
- Ticket prices

---

# 📂 Project Structure

```
flight-price-prediction
│
├── streamlit_app.py
├── app.py
├── c1_flight_rf.pkl
├── c2_flight_rf.pkl
├── requirements.txt
├── static/
├── templates/
└── README.md
```

**Description**

- `streamlit_app.py` – Main Streamlit application interface
- `app.py` – Original backend prediction logic
- `c1_flight_rf.pkl` – Trained Random Forest model
- `c2_flight_rf.pkl` – Alternative trained model
- `requirements.txt` – Project dependencies
- `static/` – Images and UI assets
- `templates/` – HTML templates from earlier implementation

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/kennyrogers330/flight-price-prediction.git
cd flight-price-prediction
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run streamlit_app.py
```

After running the command, the application will open in your browser:

```
http://localhost:8501
```

---

# 🧩 How the Prediction Works

1. The user enters travel details such as departure date, arrival time, airline, source city, destination city, and number of stops.
2. The application performs feature engineering on the input values.
3. The features are converted into numerical format using the same transformations used during model training.
4. The trained Random Forest regression model receives the processed input.
5. The application returns an estimated flight ticket price.

---

# 📸 Example Interface

The web application provides a clean interface where users can enter travel details and receive predicted ticket prices instantly.

---

---

# 👤 Author

Machine Learning Flight Fare Prediction Application developed using Python, Scikit-learn, and Streamlit.
