# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy over the dependency definitions first to optimize caching
COPY requirements.txt .

# Install the exact Python packages specified without caching overhead
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code directory structure into the container workspace
COPY . .

# Expose port 8501, which is the standard port Streamlit listens on
EXPOSE 8501

# Launch the app automatically when the container initializes
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]