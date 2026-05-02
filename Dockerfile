FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV + general ML tooling
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

COPY requirements-docker.txt ./requirements-docker.txt

# Install PyTorch CPU wheels explicitly for reliable Docker builds
RUN python -m pip install -U pip \
  && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r requirements-docker.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY tools ./tools
COPY configs ./configs
COPY outputs ./outputs

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]

