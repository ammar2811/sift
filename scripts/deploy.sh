#!/usr/bin/env bash
#
# Deploy Sift to Azure Container Apps.
#
# This runs locally rather than in CI because the Entra tenant this subscription lives
# in sets `allowedToCreateApps: false` - no app registration means no service principal
# and no OIDC federated credential, so GitHub Actions has no way to authenticate. It
# uses an interactive `az login` instead.
#
# Usage:
#   scripts/deploy.sh                 # deploy the images tagged :latest
#   scripts/deploy.sh <git-sha>       # deploy a specific immutable tag
#   scripts/deploy.sh --verify-only   # just probe the running revision
#
set -euo pipefail

RESOURCE_GROUP="${SIFT_RESOURCE_GROUP:-sift-rg}"
REGISTRY="${SIFT_REGISTRY:-ghcr.io}"
REPO="${SIFT_GITHUB_REPO:-ammar2811/sift}"
TAG="${1:-latest}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

command -v az >/dev/null || fail "az CLI is not installed"
az account show >/dev/null 2>&1 || fail "not logged in - run: az login"

api_fqdn() {
  az containerapp show --name sift-api --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null
}

verify() {
  local fqdn url
  fqdn="$(api_fqdn)" || fail "sift-api not found in $RESOURCE_GROUP"
  [ -n "$fqdn" ] || fail "sift-api has no ingress FQDN"
  url="https://$fqdn"

  # Scale-to-zero means the first request pays a cold start, so this retries rather
  # than reporting a failure that is really just a container waking up.
  log "probing $url/health"
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 15 "$url/health" >/dev/null 2>&1; then
      log "healthy after ${attempt} attempt(s)"
      echo
      curl -fsS "$url/ready" | python3 -m json.tool || true
      echo
      log "API:  $url"
      log "Web:  https://$(az containerapp show --name sift-web \
        --resource-group "$RESOURCE_GROUP" \
        --query properties.configuration.ingress.fqdn -o tsv)"
      return 0
    fi
    sleep 10
  done
  fail "the revision never became healthy"
}

if [ "$TAG" = "--verify-only" ]; then
  verify
  exit 0
fi

log "deploying tag '$TAG' from $REGISTRY/$REPO to $RESOURCE_GROUP"

log "updating sift-api"
az containerapp update \
  --name sift-api \
  --resource-group "$RESOURCE_GROUP" \
  --image "$REGISTRY/$REPO/api:$TAG" \
  --output none

log "updating sift-web"
az containerapp update \
  --name sift-web \
  --resource-group "$RESOURCE_GROUP" \
  --image "$REGISTRY/$REPO/web:$TAG" \
  --output none

# The ingestion job runs the same image as the API, only with a different entrypoint.
if az containerapp job show --name sift-ingest --resource-group "$RESOURCE_GROUP" \
    >/dev/null 2>&1; then
  log "updating sift-ingest job"
  az containerapp job update \
    --name sift-ingest \
    --resource-group "$RESOURCE_GROUP" \
    --image "$REGISTRY/$REPO/api:$TAG" \
    --output none
fi

# A deploy that reports success while the app is broken is worse than one that fails.
verify
