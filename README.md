# hungryhub_automation

This is a full-stack web application built using Flask (Python) for the backend and React (JavaScript) for the frontend.

## Features
- REST API built with Flask
- Frontend powered by React
- CORS enabled for seamless communication between frontend and backend

---

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- [Python 3](https://www.python.org/downloads/)
- [Node.js & npm](https://nodejs.org/)

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/flask-react-app.git
cd flask-react-app
```

---

## Backend Setup (Flask)

### 2. Navigate to the Backend Directory
```sh
cd backend
```

### 3. Install Dependencies INSIDE backend directory
```sh
pip install flask flask-cors flask-restful
pip install python-dotenv
pip install anthropic
pip install google-genai
```

### 4. Run the Flask Backend INSIDE backend directory
```sh
python app.py
```
Flask will start on `http://127.0.0.1:5000`.

#### API Endpoints
| Method | Endpoint      | Description             |
|--------|--------------|-------------------------|
| GET    | /api/hello   | Returns a welcome message |

---

## Frontend Setup (React)

### 5. Navigate to the Frontend Directory
```sh
cd frontend
```

### 6. Install Dependencies
```sh
npm install
npm install axios
npm install react-bootstrap bootstrap
```

### 7. Start the React App
```sh
npm start
```
The React app will be available at `http://localhost:3000`.

---

## Deployment

### 8. Build the React App
```sh
cd frontend
npm run build
```

