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
#   scripts/deploy.sh <git-sha>       # deploy a specific immutable tag, short or full
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

# The workflow tags images with ${{ github.sha }}, which is the full 40-character hash,
# while `git log --oneline` and every commit message abbreviate it. Passing the short
# form deployed a tag that does not exist: Container Apps accepted the update for the
# two apps and only rejected it on the third, leaving two apps pointing at an image
# that could never be pulled. Resolving it here means the SHA a human reads and the SHA
# the registry wants are the same argument.
if [[ "$TAG" =~ ^[0-9a-f]{7,39}$ ]]; then
  if full="$(git rev-parse "$TAG" 2>/dev/null)"; then
    [ "$full" != "$TAG" ] && log "resolved $TAG to $full"
    TAG="$full"
  else
    fail "'$TAG' looks like an abbreviated SHA but git cannot resolve it; pass the full hash"
  fi
fi

api_fqdn() {
  az containerapp show --name sift-api --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null
}

# Check the tag exists before anything is updated.
#
# Container Apps validates the image per app, at update time, so a bad tag is not
# rejected as one atomic mistake: it is accepted for sift-api, accepted for sift-web,
# and only refused for sift-ingest. That leaves the deployment half-applied and pointing
# at an image no replica can pull. One HEAD against the registry beforehand turns that
# into a failure with nothing changed.
require_image() {
  local image="$1" token url
  [ "$TAG" = "latest" ] && return 0

  token="$(curl -fsS "https://ghcr.io/token?scope=repository:${REPO}/${image}:pull" 2>/dev/null \
    | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')"
  if [ -z "$token" ]; then
    log "could not reach the registry to verify ${image}:${TAG}; continuing"
    return 0
  fi

  url="https://ghcr.io/v2/${REPO}/${image}/manifests/${TAG}"
  curl -fsS -o /dev/null -X HEAD -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.oci.image.index.v1+json" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
    "$url" 2>/dev/null \
    || fail "no image ${REGISTRY}/${REPO}/${image}:${TAG} - has CI finished publishing it?"
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

require_image api
require_image web

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
