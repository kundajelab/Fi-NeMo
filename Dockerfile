# Use the official Miniconda3 image as our base
FROM continuumio/miniconda3:latest

# Install git (and any other system packages if needed)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory where the repository will be cloned
WORKDIR /opt

# Clone the Fi-NeMo repository from GitHub
RUN git clone https://github.com/kundajelab/Fi-NeMo.git

# Change directory into the repository
WORKDIR /opt/Fi-NeMo

# Create a conda environment named "finemo" from the environment.yml file
RUN conda env create -f environment.yml -n finemo

# Ensure that commands run in the container use the new conda environment
# by prepending its bin directory to the PATH.
ENV PATH /opt/conda/envs/finemo/bin:$PATH

# Install the Fi-NeMo package in editable mode.
RUN pip install --editable .

# (Optional) Set an entrypoint or default command.
# Here we simply open a bash shell. You can override this when running the container.
CMD ["bash"]
