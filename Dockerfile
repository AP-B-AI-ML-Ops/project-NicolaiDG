# Use the Anaconda image from Microsoft Container Registry
FROM mcr.microsoft.com/devcontainers/anaconda:0-3

# Copy environment.yml (if found) to update the environment.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Set the working directory in the container
WORKDIR /workspace/project-NicolaiDG

# Copy the current directory contents into the container
COPY . /workspace/project-NicolaiDG

# Install any Python dependencies specified in requirements.txt
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Run db.py when the container launches
CMD ["python", "db.py"]
