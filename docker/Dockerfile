FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (less frequent changes)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy project definition and metadata files for dependency installation
# These change less frequently than source code and are used for the dependency cache layer.
COPY pyproject.toml ./
COPY README.md ./

# Create a minimal src structure for the initial pip install to succeed
# This allows pip/setuptools to discover the package and install dependencies.
RUN mkdir -p src && touch src/__init__.py

# Install Python dependencies (and a stub of the project).
# This layer is cached if pyproject.toml, README.md, and the minimal src structure don't change.
RUN pip install --no-cache-dir --break-system-packages .

# Copy the actual application source code.
# This layer changes more frequently. Overwrites the minimal /app/src.
COPY src /app/src

# Re-install the project using the full source code.
# Dependencies are already cached from the previous pip install.
# This ensures the correct version of the project is installed in site-packages.
RUN pip install --no-cache-dir --break-system-packages .

# Define the entrypoint.
# The project 'repo_explainer_agent' (mapped from /app/src) is already "installed"
# by the 'pip install .' step above. Python will find the full code copied in the previous step.
ENTRYPOINT ["python", "-m", "repo_explainer_agent.graph"]
