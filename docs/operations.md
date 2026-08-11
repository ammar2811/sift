# Operations

Deploying, rolling back, and reading a failure. For what the system is and how well it
works, see the [README](../README.md); for why the numbers are what they are, see
[eval/results](../eval/results/README.md).

## Deploying

CI publishes an image per commit to GHCR, tagged with the git SHA and with `latest`.
Rolling one out is a local command:

```bash
az login
./scripts/deploy.sh                # whatever is tagged latest
./scripts/deploy.sh 8ddb892        # a specific commit, which is what you want
./scripts/deploy.sh --verify-only  # check what is running without changing it
```

Deploying by SHA rather than by `latest` is the difference between a deployment you can
reason about and one you cannot: `latest` moves under you, so two replicas started
minutes apart can be running different code.

The script polls `/health` for up to five minutes before reporting success, because the
app scales to zero and the first request after a deploy pays for a cold start. It then
prints `/ready`, which is the check that actually tells you whether the app can serve.

**Deploying from CI is not possible on this subscription**, and this is a tenant policy
rather than an omission: the Entra directory sets `allowedToCreateApps: false`, so no
app registration can be created, so there is no service principal and no OIDC federated
credential for GitHub Actions to authenticate with. If that policy changes, the deploy
job is a dozen lines and the images are already published.

## Rolling back

There is no rollback script because there does not need to be one. Images are immutable
and tagged by SHA:

```bash
./scripts/deploy.sh <previous-good-sha>
```

`git log --oneline` on `main` gives the candidates. Container Apps also keeps revisions,
so `az containerapp revision list -n sift-api -g sift-rg` shows what has run and
`az containerapp ingress traffic set` can shift traffic back without a new deploy.

## Infrastructure changes

```bash
cp infra/params.example.json infra/params.json   # then fill in the secrets
az deployment group create -g sift-rg -f infra/main.bicep -p @infra/params.json
```

The template is idempotent and safe to re-run. It does **not** create the Azure OpenAI
account: that lives in East US where the model quota is, was created by hand, and is
passed in as an endpoint and a key. The application stack is in Canada Central because
PostgreSQL Flexible Server provisioning is restricted for student subscriptions in East
US, East US 2 and West US 2.

## Reading a failure

`/ready` names its own cause. Check it first.

```bash
curl -s https://<api-fqdn>/ready | jq
```

| what it says | what happened |
|---|---|
| `postgres: ok=false` | The database is unreachable or the pool is exhausted. `SIFT_DB_POOL_MAX` is 5 per replica against a B1ms; `/api/ask` holds a connection for the life of its stream, so a burst of questions can consume the pool. |
| `embeddings: ok=false`, mentions dimensions | The provider produces vectors of a different width than the corpus was ingested at. pgvector refuses the comparison, so every query fails. Point `SIFT_EMBEDDING_PROVIDER` at the provider used for ingestion, or re-ingest. |
| `chat: not configured` | `/api/ask` returns 503; search still works. Azure OpenAI credentials are missing from the container. |
| `redis: ok=false` | Not fatal. The embedding cache is skipped and queries get slower - see [load/README.md](../load/README.md) for how much. |
| `ready: false`, `chunks: 0` | No active corpus version. Ingestion has not run, or no version was activated. |

`/health` is deliberately different: it touches nothing and answers as long as the
process is alive. It is what the orchestrator restarts on, so it must not fail when the
database blips or a database outage becomes a restart storm.

## Logs and traces

Logs are JSON, one object per line, every line carrying `request_id`. In Log Analytics:

```kusto
ContainerAppConsoleLogs_CL
| where Log_s has "request_id"
| extend p = parse_json(Log_s)
| where p.path == "/api/ask"
| project TimeGenerated, p.request_id, p.status, p.duration_ms
| order by TimeGenerated desc
```

A client can pass `X-Request-ID` and it will be honoured and echoed back, so a request
can be followed from the browser through nginx into the API.

Traces go to Application Insights when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set,
which the Bicep template does. If the package is missing or the string is absent the app
runs untraced rather than failing to start.

## Re-ingesting the corpus

Ingestion is queue-driven and idempotent - chunks are replaced per document in one
transaction, so a retry converges rather than duplicating.

```bash
python -m apps.worker.fetch_corpus          # ~1,700 documents, throttled, resumable
python -m apps.worker.enqueue --all         # creates the corpus version, then enqueues
```

KEDA starts the ingest job on queue depth and it exits when the queue drains. A new
corpus version is inert until activated, so a rebuild does not disturb what is serving:

```bash
python -m apps.worker.ingest --activate     # switch traffic to the new version
```

Changing the embedding width means a full re-ingest. `/ready` will refuse to report
ready if the running provider disagrees with the active corpus, which is the guard
against half-switching.

## Costs

Sized for a student subscription's $100 ceiling. Postgres B1ms is the standing cost;
Container Apps and the ingest job scale to zero. Redis is roughly $14/month and is on
because it is [measurably worth it](../load/README.md#what-redis-is-actually-worth)
across the restarts that scale-to-zero causes. Answers cost about $0.0028 each, which is
why `/api/ask` carries a per-client rate limit and search does not.
