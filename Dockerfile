# sky-clock with local plate solving (astrometry.net + Tycho-2 wide-field index files, ~10-90 deg fields)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    SKYCLOCK_CACHE=/data/cache
RUN apt-get update && apt-get install -y --no-install-recommends \
      astrometry.net astrometry-data-tycho2-10-19 \
      python3 python3-pip python3-venv fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /venv && /venv/bin/pip install -r requirements.txt
ENV PATH="/venv/bin:$PATH"
COPY . .
RUN pip install --no-deps -e .

EXPOSE 8501
VOLUME ["/data"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
