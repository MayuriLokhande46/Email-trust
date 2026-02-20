# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Define environment variable
ENV PYTHONPATH=/app/spam_detector/src

# Create a script to run both services (super minimal approach for single container, 
# though docker-compose is better for separation, we provide a default command)
CMD ["uvicorn", "spam_detector.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
