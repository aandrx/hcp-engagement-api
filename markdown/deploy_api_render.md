# Deploying an API to Render (Free Tier)

This guide walks you through deploying a backend API to [Render](https://render.com).

---

## 1. Prepare Your Project

1. Push your project to **GitHub** (Render connects directly to GitHub).
2. Add a proper start command for your API:

   - **Node.js (Express)**: in `package.json`
     ```json
     "scripts": {
       "start": "node index.js"
     }
     ```

   - **Python (Flask)**:
     - Add `requirements.txt`.
     - Create a `Procfile`:
       ```txt
       web: gunicorn app:app
       ```
       *(Assuming `app.py` contains `app = Flask(__name__)`).*

3. Ensure your API binds to the **Render-provided port**:
   ```js
   const PORT = process.env.PORT || 10000;
   app.listen(PORT, () => console.log(`Server running on ${PORT}`));
   ```

---

## 2. Create a Render Account & Connect GitHub

1. Sign up at [Render](https://render.com) using **GitHub login**.
2. Authorize access so Render can read your repos.

---

## 3. Create a New Web Service

1. Go to the Render dashboard → **New → Web Service**.
2. Select your GitHub repo.
3. Configure:
   - **Name** → e.g., `my-api` (URL will be `my-api.onrender.com`).
   - **Region** → closest to your users.
   - **Branch** → `main`.
   - **Build Command**:
     - Node.js → `npm install`
     - Python → `pip install -r requirements.txt`
   - **Start Command**:
     - Node.js → `npm start`
     - Python → `gunicorn app:app`

---

## 4. Deploy

1. Click **Create Web Service**.
2. Render builds and starts your app automatically.
3. Your API is now live at:
   ```
   https://your-api.onrender.com
   ```

---

## 5. Test Your API

Use `curl` or a browser:
```bash
curl https://your-api.onrender.com/your-endpoint
```

---

## 6. Redeploys & Updates

- Every push to GitHub triggers an automatic redeploy.
- You can manually redeploy from the Render dashboard.

---

## 7. Free Tier Notes

- 750 free instance hours/month → one always-on API.
- Services sleep after 15 min of inactivity (wake on request).
- Free SSL certificate and custom domain support.

---

✅ Done! Your API is deployed for free on Render.
