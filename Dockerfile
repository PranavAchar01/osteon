FROM python:3.11-slim-bookworm

# --- system libraries: headless Blender 5.x + scientific stack (vtk/open3d/pymeshlab) runtime ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget xz-utils ca-certificates \
        build-essential gfortran cmake ninja-build \
        libgl1 libegl1 libglib2.0-0 libgomp1 \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxrandr2 \
        libxext6 libxt6 libsm6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# --- Blender 5.1.2 (matches the local toolchain; used headless via --background) ---
ENV BLENDER_VERSION=5.1.2
RUN wget -q "https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -O /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz
ENV OSTEON_BLENDER=/opt/blender/blender

WORKDIR /app

# --- python deps: CPU-only torch first (slim image), then the pinned stack ---
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- app source (split_a/b/c + common import as top-level packages from /app) ---
COPY . .

ENV OSTEON_HOST=0.0.0.0 \
    PORT=8080 \
    OSTEON_TRACE_DIR=/tmp/traces \
    PYTHONUNBUFFERED=1
EXPOSE 8080

# Blender renders run as subprocesses; gthread workers keep the request alive meanwhile.
CMD ["gunicorn", "-k", "gthread", "-w", "2", "--threads", "4", "-t", "600", "-b", "0.0.0.0:8080", "webapp.app:app"]
