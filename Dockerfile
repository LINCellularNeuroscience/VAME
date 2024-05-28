# Dockerfile for VAME pipeline
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /vame

# Copy the vame package to the working directory
COPY requirements.txt /vame
COPY pyproject.toml /vame
COPY src/ /vame/src

# Copy the demo docker example to working dir
COPY ./examples/demo_docker.py /vame

# Install vame
RUN pip install --no-cache-dir .


CMD ["python", "demo_docker.py"]



