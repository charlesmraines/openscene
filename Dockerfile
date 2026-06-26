# 1. Start from your existing working MinkowskiEngine image
FROM minkowski_engine

# 2. Install system tools needed for CLIP (git) and OpenCV/Open3D (graphics libraries)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy your requirements file from your host machine into the image
COPY requirements.txt /tmp/requirements.txt

# 4. Install all the Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt