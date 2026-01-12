# MIRAI Virtual Try-On Startup Guide

This guide explains how to run the VTON Engine (Backend) and the Smart Mirror (Frontend) simultaneously.

## Prerequisites
- **Node.js**: v18 or later.
- **Python**: 3.10 recommended.
- **API Key**: Ensure you have a valid Segmind API key if `USE_PAID_API` is set to `true` in `backend/server.js`.

## 1. Running the Backend (VTON Engine)
The backend is a Node.js server that wraps the Python VTON logic.

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies (if not already done):
   ```bash
   npm install
   ```
3. Start the server:
   ```bash
   node server.js
   ```
   The backend will be listening on `http://localhost:5000`.

## 2. Running the Frontend (Smart Mirror)
The frontend is a Next.js application.

1. Open a **new** terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will be accessible at `http://localhost:3000`.

## 3. Using the Application
1. Open your browser to `http://localhost:3000`.
2. Follow the on-screen instructions:
   - Capture your photo.
   - Select a garment from the catalog.
   - Click **GENERATE TRY-ON**.
3. The results will be displayed once the backend processes the request (this may take 20-30 seconds depending on the model).
