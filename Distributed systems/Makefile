.PHONY: build up down logs test analysis clean

# Build both Docker images (server + load balancer)
build:
	docker build -t lb-server:latest ./server
	docker compose build

# Bring the whole stack up (load balancer will spawn N=3 server replicas)
up: build
	docker compose up -d
	@echo "Load balancer available at http://localhost:5000"

# Tear the stack down and remove any spawned server containers
down:
	docker compose down
	-docker ps -aq --filter "network=net1" | xargs -r docker rm -f

# Tail load balancer logs
logs:
	docker logs -f load_balancer

# Run unit tests (consistent hash map logic; no Docker required)
test:
	python3 -m pytest tests/ -v

# Run the Task 4 analysis suite against a running stack (requires `make up` first)
analysis:
	python3 analysis/run_all.py

clean: down
	docker rmi -f lb-loadbalancer:latest lb-server:latest || true
