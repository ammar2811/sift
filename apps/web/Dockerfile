# Two stages: build the bundle with Node, serve the static output with nginx.
#
#   docker build -f apps/web/Dockerfile -t sift-web apps/web

FROM node:22-alpine AS build

WORKDIR /app
RUN corepack enable

COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
# The nginx image runs envsubst over /etc/nginx/templates/*.template at start, so the
# API origin is injected per environment instead of being baked into the image.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Overridden at deploy time; this default keeps `docker compose` working locally.
ENV SIFT_API_ORIGIN=http://api:8000

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
