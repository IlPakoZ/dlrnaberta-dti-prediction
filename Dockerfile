FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install git for requirements.txt
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy the repository into the container
COPY . /workspace/dlrnaberta-dti-prediction

# Set working directory
WORKDIR /workspace/dlrnaberta-dti-prediction	

# Install Python dependencies
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]

