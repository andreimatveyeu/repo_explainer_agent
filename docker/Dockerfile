FROM python:3.12-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
# Install dependencies based on pyproject.toml
# This layer is cached and only re-runs if pyproject.toml changes.
RUN pip install --no-cache-dir --break-system-packages .

COPY README.md ./
COPY repo_explainer_agent /app/repo_explainer_agent
# Copy the rest of the application code.
# This layer re-runs if README.md or any file in repo_explainer_agent changes.

# Install the project itself, using the cached dependencies.
# This layer re-runs if the application code (copied above) changes.
RUN pip install --no-cache-dir --break-system-packages . --no-deps

ENTRYPOINT ["python", "-m", "repo_explainer_agent.repo_explainer_agent"]
