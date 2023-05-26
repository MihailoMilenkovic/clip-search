# Use the official Python base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Run scripts to index images and text
RUN python image_database.py
RUN python text_database.py
# Copy the output directory to the container
COPY output_directory1 /app/output_directory1
COPY output_directory2 /app/output_directory2


# Expose the necessary port
EXPOSE 8050

# Run the application
CMD ["python", "server.py"]