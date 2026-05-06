FROM python:3.12-slim

# Install Node.js 20 for the claude CLI
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install claude CLI
RUN npm install -g @anthropic-ai/claude-code

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /var/log/claude-api

EXPOSE 8731

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8731", "--workers", "1"]
