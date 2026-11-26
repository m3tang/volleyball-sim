#!/usr/bin/env bash
# Deploy the public "Ask the model" API to Google Cloud Run.
#
#   PROJECT=my-gcp-project serve/deploy_cloudrun.sh          # first deploy, and every update
#
# Needs: gcloud (logged in: gcloud auth login), Docker running, data/vb.duckdb, and the API key in the
# macOS Keychain as "vb-sim-agent-api-key". The key goes Keychain -> Secret Manager through a pipe, so it
# is never printed or written to shell history. Cloud Run reads it at startup as ANTHROPIC_API_KEY.
#
# Limits: one instance at most (the rate limit and daily budget are kept in memory), scale to zero when
# idle, 90 s request timeout. Change the budget or rate with BUDGET / RATE below and rerun.
set -euo pipefail

: "${PROJECT:?set PROJECT to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-rally-agent}"
SECRET="${SECRET:-vb-sim-agent-api-key}"
BUDGET="${BUDGET:-1}"
RATE="${RATE:-10}"
ORIGINS="${ORIGINS:-https://m3tang.github.io}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/rally/$SERVICE:$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

cd "$(dirname "$0")/.."
gcloud config set project "$PROJECT" >/dev/null

echo "== enabling services (first run only takes a minute)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

echo "== image registry"
gcloud artifacts repositories describe rally --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create rally --repository-format docker --location "$REGION"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

echo "== API key secret"
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  security find-generic-password -s "$SECRET" -w | tr -d '\n' | gcloud secrets create "$SECRET" --data-file=-
fi
NUMBER=$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')
gcloud secrets add-iam-policy-binding "$SECRET" --quiet >/dev/null \
  --member "serviceAccount:$NUMBER-compute@developer.gserviceaccount.com" --role roles/secretmanager.secretAccessor

echo "== build and push $IMAGE"
.venv/bin/python -m pipeline.public_db
docker build --platform linux/amd64 -f serve/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

echo "== deploy"
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --allow-unauthenticated \
  --min-instances 0 --max-instances 1 --concurrency 8 --cpu 1 --memory 512Mi --timeout 90 \
  --set-secrets "ANTHROPIC_API_KEY=$SECRET:latest" \
  --set-env-vars "^@^DAILY_BUDGET_USD=$BUDGET@RATE_PER_HOUR=$RATE@ALLOWED_ORIGINS=$ORIGINS"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "== live at $URL"
curl -s "$URL/health"; echo
