# 1. Start from a standard Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set the PORT environment variable
# Cloud Run will use this to run your app
ENV PORT 8080

# 4. Copy *only* the requirements file first to cache this layer
COPY requirements.txt .

# 5. Install your Python libraries
# Using --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your *entire* project into the container
# This includes app.py, your .keras file (or model folder),
# and the /templates folder
COPY . .

# 7. Tell Cloud Run how to start your app
# This command runs gunicorn, which serves your 'app' variable
# from your 'app.py' file.
CMD gunicorn --bind 0.0.0.0:$PORT app:app
