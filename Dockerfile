# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the entire current directory into the container
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the server will run
EXPOSE 8050

# Set the command to run the server when the container starts
CMD ["python", "server.py"]