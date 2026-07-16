import os
from flask import Flask, jsonify

app = Flask(__name__)

# Server ID is passed in as an environment variable when the container is
# started (see Dockerfile / docker-compose.yml / load balancer spawn logic).
SERVER_ID = os.environ.get("SERVER_ID", "unknown")


@app.route("/home", methods=["GET"])
def home():
    return jsonify({
        "message": f"Hello from Server: {SERVER_ID}",
        "status": "successful"
    }), 200


@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    # Empty body, 200 status code is enough to signal "I'm alive".
    return "", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
