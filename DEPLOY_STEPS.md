# 🚀 Deployment Guide: Hosting Your Pneumonia Detection App

This guide explains how to deploy this project online so anyone can use it.

---

## 💡 The Core Problem with Vercel for this Project
- **Vercel** is a serverless platform designed for light APIs and frontend websites.
- Vercel strictly enforces a **250 MB maximum bundle limit**.
- **PyTorch + TorchVision alone are ~1 GB**, plus your trained ResNet50 model weights (`best_pneumonia_model.pth`) are **94 MB**.
- If deployed directly to Vercel, the build will fail immediately with: `Serverless Function exceeds 250 MB`.

---

## ✅ The Best Solution: Host on Render.com (100% Free & Works Out of the Box)
Render is a cloud container platform that supports full Python environments, PyTorch, and large models. Both your web UI and deep learning model will run together without serverless size limits.

### Step-by-Step Instructions:

### 1️⃣ Push to GitHub
Open your terminal in this folder and initialize/push to your GitHub repository:
```bash
git init
git add .
git commit -m "Prepare Pneumonia Detection app for cloud deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
*(Note: `.gitignore` will ensure your huge `chest_xray/` dataset and `venv/` are NOT pushed, keeping your repo clean and under GitHub limits).*

---

### 2️⃣ Deploy on Render.com
1. Go to [https://render.com](https://render.com) and Sign In (or Sign Up with GitHub).
2. Click **New +** (top right) and choose **Web Service**.
3. Select **Build and deploy from a Git repository**.
4. Choose your GitHub repository.
5. Fill in the deployment settings:
   - **Name**: `pneumonia-detection` (or any name you like)
   - **Region**: Choose the closest region (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app`
   - **Instance Type**: **Free**
6. Under **Environment Variables**, add these (recommended):
   - `PYTHON_VERSION` = `3.11.9`
   - `FLASK_ENV` = `production`
   - `OMP_NUM_THREADS` = `1`
   - `MKL_NUM_THREADS` = `1`
7. Click **Deploy Web Service** at the bottom.

Render will install dependencies, load the ResNet50 model, and give you a live HTTPS URL:
`https://pneumonia-detection-xxxx.onrender.com`

---

## 🌐 What if you STILL want a Vercel URL? (Vercel Proxy Method)
If you want your website to open from a `.vercel.app` link, you can use Vercel as a **Front-End Proxy**:
1. Deploy the backend to Render as shown above (e.g. `https://my-backend.onrender.com`).
2. Add a `vercel.json` file in a Vercel project:
```json
{
  "version": 2,
  "rewrites": [
    { "source": "/(.*)", "destination": "https://my-backend.onrender.com/$1" }
  ]
}
```
3. Deploy to Vercel. Now users visit your Vercel link, and Vercel transparently forwards requests to your Render AI backend!

