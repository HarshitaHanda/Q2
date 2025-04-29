# Lab-Report Extractor API

FastAPI service that extracts lab‑test names, obtained values, units and reference ranges from diagnostic report images or single‑page PDFs.

---

## Local quick‑start

```bash
# 1 .  create / activate virtualenv (optional)
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# 2 .  install deps
pip install -r requirements.txt

# 3 .  run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Swagger UI** will be available at **http://localhost:8000/docs**

Upload a PNG / JPEG / single‑page PDF in the UI to get structured JSON like:

```json
{
  "is_success": true,
  "data": {
    "lab_tests": [
      {
        "test_name": "HAEMOGLOBIN",
        "obtained_value": "13.4",
        "bio_reference_range": "12.0-16.0",
        "unit": "g/dl",
        "lab_test_out_of_range": false
      }
    ]
  }
}
```

---

## Docker

```bash
docker build -t lab-extractor .
docker run -p 8000:8000 lab-extractor
```

Then open **http://localhost:8000/docs**.

---

## Deployments

* **Render / Railway (Docker)** – push repo, select *Docker* environment, no extra settings.
* **Cloudflare Tunnel / ngrok** – expose `localhost:8000` for quick external demos.

---

## Endpoints

| Method | Path            | Purpose                               |
|--------|-----------------|---------------------------------------|
| POST   | `/get-lab-tests`| Accepts an image/PDF; returns JSON    |
| GET    | `/docs`         | Interactive Swagger documentation UI |

---

## License

MIT

