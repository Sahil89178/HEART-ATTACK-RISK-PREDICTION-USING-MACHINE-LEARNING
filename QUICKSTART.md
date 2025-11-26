# Quick Start Guide

Follow these steps to get your Heart Attack Prediction application up and running quickly.

## Step 1: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up MongoDB

### Option A: Local MongoDB
1. Install MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Start MongoDB service
3. Default connection: `mongodb://localhost:27017/heart_prediction`

### Option B: MongoDB Atlas (Cloud)
1. Create a free account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a cluster and get your connection string
3. Update `MONGO_URI` in `app.py` or set as environment variable

## Step 3: Start Backend Server

```bash
python app.py
```

The server will run on `http://localhost:5000`

## Step 4: Install Frontend Dependencies

Open a new terminal and navigate to the client folder:

```bash
cd client
npm install
```

## Step 5: Start Frontend Development Server

```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

## Step 6: Access the Application

1. Open your browser and go to `http://localhost:3000`
2. Click "Sign up" to create a new account
3. Fill in your details and register
4. You'll be automatically logged in and redirected to the dashboard
5. Start making predictions!

## Troubleshooting

### MongoDB Connection Error
- Make sure MongoDB is running (if using local)
- Check your connection string in `app.py`
- Verify MongoDB port (default: 27017)

### Port Already in Use
- Backend: Change port in `app.py` (last line)
- Frontend: Vite will automatically use the next available port

### Module Not Found Errors
- Make sure you've installed all Python dependencies: `pip install -r requirements.txt`
- Make sure you've installed all Node dependencies: `cd client && npm install`

### CORS Errors
- Ensure backend is running on port 5000
- Check that `flask-cors` is installed

## Next Steps

- Make your first prediction on the Dashboard
- View your prediction history in Records
- Update your profile information
- Toggle between dark and light mode using the theme button

Enjoy using HeartAI! 🚀

