FROM python:3.11-slim

WORKDIR /app

# Docker CLI is not strictly required since we use the Docker SDK (talking
# directly to the socket), but curl is handy for debugging inside the
# container.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py consistent_hash.py ./

ENV N=3
ENV NUM_SLOTS=512
ENV NUM_VIRTUAL_SERVERS=9
ENV DOCKER_NETWORK=net1
ENV SERVER_IMAGE=lb-server:latest

EXPOSE 5000

CMD ["python", "app.py"]
