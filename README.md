# Heart Attack Risk Prediction System

A modern, full-stack web application for predicting heart attack risk using machine learning, with user authentication, MongoDB data storage, and a beautiful responsive UI.

## Features

- 🧠 **Machine Learning Prediction**: Neural network-based heart attack risk prediction
- 🔐 **User Authentication**: Secure JWT-based authentication system
- 📊 **Dashboard**: Interactive dashboard with prediction form and statistics
- 📝 **Prediction Records**: View and manage your complete prediction history
- 👤 **User Profile**: Manage your account information and settings
- 🌙 **Dark/Light Mode**: Toggle between dark and light themes
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- 🗄️ **MongoDB Integration**: Persistent data storage for users and predictions

## Tech Stack

### Backend
- Flask (Python web framework)
- PyTorch (Neural network model)
- MongoDB (Database)
- JWT (Authentication)
- Flask-CORS (Cross-origin support)

### Frontend
- React 18
- React Router (Navigation)
- Tailwind CSS (Styling)
- Vite (Build tool)
- Axios (HTTP client)
- Lucide React (Icons)

## Project Structure

```
Heart Attack Prediction/
├── app.py                 # Flask backend server
├── models.py              # MongoDB models (User, PredictionRecord)
├── nn_model.py           # Neural network model definition
├── train_nn.py           # Training script (unchanged)
├── train_rf.py           # Random forest training (unchanged)
├── models/               # Saved ML models
│   ├── nn_model.pth
│   ├── rf_model.pkl
│   └── scaler.pkl
├── requirements.txt      # Python dependencies
├── client/              # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   ├── context/     # React context providers
│   │   ├── utils/       # Utility functions
│   │   ├── App.jsx      # Main app component
│   │   └── main.jsx     # Entry point
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local installation or MongoDB Atlas)

### 1. Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up MongoDB:**
   - Install MongoDB locally, or
   - Use MongoDB Atlas (cloud)
   - Update the connection string in `app.py` if needed (default: `mongodb://localhost:27017/heart_prediction`)

3. **Set environment variables (optional):**
   ```bash
   export JWT_SECRET_KEY="your-secret-key-here"
   export MONGO_URI="mongodb://localhost:27017/heart_prediction"
   ```

4. **Run the Flask server:**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

### 2. Frontend Setup

1. **Navigate to the client directory:**
   ```bash
   cd client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   The frontend will start on `http://localhost:3000`

### 3. Access the Application

- Open your browser and navigate to `http://localhost:3000`
- Register a new account or login
- Start making predictions!

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user info

### Predictions
- `POST /api/predict` - Make a prediction (requires authentication)
- `GET /api/records` - Get all user predictions
- `DELETE /api/records/<id>` - Delete a prediction record

### Profile
- `PUT /api/profile` - Update user profile

## ML Model Details

The application uses a neural network model trained on heart disease data. The model takes 11 input features:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

The model outputs a probability score, which is converted to "High Risk" or "Low Risk" classification.

**Note**: The existing ML training code (`train_nn.py`, `train_rf.py`) remains completely unchanged as requested.

## Environment Variables

You can set these environment variables for production:

- `JWT_SECRET_KEY`: Secret key for JWT token signing
- `MONGO_URI`: MongoDB connection string

## Development

### Backend Development
- The Flask server runs in debug mode by default
- API endpoints are available at `http://localhost:5000/api/*`

### Frontend Development
- Hot module replacement is enabled
- Proxy configured to forward `/api` requests to Flask backend

## Production Deployment

1. **Build the frontend:**
   ```bash
   cd client
   npm run build
   ```

2. **Serve static files:**
   - Configure Flask to serve the built React app, or
   - Use a web server (Nginx, Apache) to serve the frontend

3. **Set production environment variables:**
   - Use a strong `JWT_SECRET_KEY`
   - Configure MongoDB connection string
   - Set `FLASK_ENV=production`

## Security Notes

- Passwords are hashed using Werkzeug's password hashing
- JWT tokens are used for authentication
- CORS is configured for the React frontend
- API routes are protected with `@jwt_required()` decorator

## License

This project is for educational purposes.

## Support

For issues or questions, please check the code comments or create an issue in the repository.

