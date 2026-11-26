# Public "Ask the model" API. Build from the repo root:
#   python -m pipeline.public_db                       # writes serve/vb_public.duckdb (needs data/vb.duckdb)
#   docker build -f serve/Dockerfile -t rally-agent .
#   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... rally-agent
FROM python:3.12-slim

WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

COPY vbsim/__init__.py vbsim/data.py vbsim/matchsim.py vbsim/
COPY rallytools/__init__.py rallytools/core.py rallytools/sql.py rallytools/grounding.py rallytools/registry.py rallytools/
COPY rallyagent/__init__.py rallyagent/agent.py rallyagent/api.py rallyagent/pricing.py rallyagent/prompts.py rallyagent/tools.py rallyagent/
COPY web/data.json web/data.json
COPY serve/vb_public.duckdb serve/vb_public.duckdb

ENV RALLY_DB=/app/serve/vb_public.duckdb PYTHONUNBUFFERED=1 PORT=8080
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn rallyagent.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
