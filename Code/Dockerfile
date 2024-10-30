# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /usr/SalesPrediction/app

# Copy all the contents of the current directory to the working directory
COPY . /usr/SalesPrediction/app

# Expose the port for Flask
EXPOSE 5000

# Install required build dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run the Flask app
CMD ["python", "app.py"]
