// Sift infrastructure.
//
// Sized for an Azure for Students subscription: $100 of credit for twelve months,
// with a hard stop and no overage. Every choice below is shaped by that ceiling.
//
//   az deployment group create -g sift-rg -f infra/main.bicep -p @infra/params.json

targetScope = 'resourceGroup'

// Canada Central, not East US. PostgreSQL Flexible Server reports "Provisioning is
// restricted in this region" for East US on this subscription - a restriction Azure
// applies to student and free subscriptions - and East US 2 and West US 2 are
// restricted too. Canada Central offers Burstable B1ms and Postgres 16, and is the
// closest region to Toronto, so it wins on latency as well as availability.
//
// The Azure OpenAI account stays in East US, where the model quota lives. That costs
// one cross-region API call per request, against a per-query cost if the database
// were remote instead.
@description('Deployment region for the application stack.')
param location string = 'canadacentral'

@description('Prefix for every resource name.')
param namePrefix string = 'sift'

@description('Container image for the API and the ingestion job.')
param apiImage string = 'ghcr.io/ammar2811/sift/api:latest'

@description('Container image for the web frontend.')
param webImage string = 'ghcr.io/ammar2811/sift/web:latest'

@description('Administrator login for PostgreSQL.')
param postgresAdminUser string = 'sift'

@secure()
@description('Administrator password for PostgreSQL.')
param postgresAdminPassword string

@secure()
@description('Azure OpenAI API key. Empty falls back to the local embedding provider.')
param azureOpenAiKey string = ''

@description('Azure OpenAI endpoint, e.g. https://sift-openai.openai.azure.com/')
param azureOpenAiEndpoint string = ''

@description('Run Redis continuously. An always-on 0.25 vCPU replica is roughly $14/month. On by default now that load/ measures what the cache is worth; turn it off again if the numbers do not justify it.')
param enableRedis bool = true

@description('Questions per minute per client on /api/ask, the only endpoint that spends money. 0 disables the limiter. Counted per replica - see apps/api/ratelimit.py.')
param askRateLimitPerMinute int = 10

var tags = {
  project: 'sift'
  managedBy: 'bicep'
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${namePrefix}-logs'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: {
      // 5 GB per month is free; capping keeps a logging loop from quietly eating
      // the credit that the rest of the project depends on.
      dailyQuotaGb: json('0.16')
    }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${namePrefix}-insights'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// ---------------------------------------------------------------------------
// Storage — the ingestion queue
// ---------------------------------------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: '${namePrefix}stor${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
  }
}

resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource ingestQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = {
  parent: queueService
  name: 'sift-ingest'
}

// ---------------------------------------------------------------------------
// PostgreSQL with pgvector
// ---------------------------------------------------------------------------

// B1ms is in the Azure for Students free-services list: 750 hours per month and
// 32 GB of storage for twelve months. It has one vCore and 2 GiB of memory, which is
// why vectors are stored as halfvec and truncated to 768 dimensions.
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: '${namePrefix}-pg'
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    administratorLogin: postgresAdminUser
    administratorLoginPassword: postgresAdminPassword
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

resource postgresDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: 'sift'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// pgvector has to be allow-listed on the server before CREATE EXTENSION will work.
resource pgvectorAllowlist 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = {
  parent: postgres
  name: 'azure.extensions'
  properties: {
    value: 'VECTOR'
    source: 'user-override'
  }
}

resource allowAzureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ---------------------------------------------------------------------------
// Container Apps
// ---------------------------------------------------------------------------

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var databaseUrl = 'postgresql://${postgresAdminUser}:${postgresAdminPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/sift?sslmode=require'
var redisUrl = enableRedis ? 'redis://${namePrefix}-redis:6379/0' : ''

resource redis 'Microsoft.App/containerApps@2024-03-01' = if (enableRedis) {
  name: '${namePrefix}-redis'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: false
        targetPort: 6379
        transport: 'tcp'
      }
    }
    template: {
      containers: [
        {
          name: 'redis'
          image: 'redis:7-alpine'
          // An in-memory cache with an eviction policy, not a datastore; nothing here
          // is worth persisting across a restart.
          args: ['--save', '', '--appendonly', 'no', '--maxmemory', '200mb', '--maxmemory-policy', 'allkeys-lru']
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        // A cache that scales to zero is not a cache.
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-api'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      secrets: [
        { name: 'database-url', value: databaseUrl }
        { name: 'azure-openai-key', value: azureOpenAiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'SIFT_DATABASE_URL', secretRef: 'database-url' }
            { name: 'SIFT_REDIS_URL', value: redisUrl }
            { name: 'SIFT_AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
            { name: 'SIFT_AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-key' }
            { name: 'SIFT_EMBEDDING_PROVIDER', value: empty(azureOpenAiKey) ? 'local' : 'azure_openai' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            // B1ms allows few connections; a large pool per replica exhausts them.
            // /api/ask holds one for the life of the stream, so this also bounds
            // concurrent questions per replica.
            { name: 'SIFT_DB_POOL_MAX', value: '5' }
            { name: 'SIFT_ASK_RATE_LIMIT_PER_MINUTE', value: string(askRateLimitPerMinute) }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 30
            }
            {
              // Readiness checks dependencies, so a database outage removes the
              // replica from rotation without restarting it.
              type: 'Readiness'
              httpGet: { path: '/ready', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 15
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        // Scale to zero: the free grant covers 180,000 vCPU-seconds a month, and a
        // portfolio URL is idle nearly all of the time. The cost is a cold start of a
        // few seconds on the first request.
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-concurrency'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

resource web 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-web'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'web'
          image: webImage
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            // Where nginx proxies /api to. This has to be here rather than set by hand:
            // the image defaults it to http://api:8000 for docker compose, that name does
            // not resolve in Container Apps, and nginx aborts at startup rather than
            // starting degraded. It had been set manually on the app, so the first
            // deployment of this template silently removed it and the web revision failed
            // to start - the previous revision kept serving, which is the only reason
            // that was not an outage.
            {
              name: 'SIFT_API_ORIGIN'
              value: 'https://${api.properties.configuration.ingress.fqdn}'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}

// Ingestion is bursty and rare. As an event-driven job it starts on queue depth and
// returns to zero when drained, so it costs nothing at rest while still being a
// genuinely queue-decoupled worker.
resource ingestJob 'Microsoft.App/jobs@2024-03-01' = {
  name: '${namePrefix}-ingest'
  location: location
  tags: tags
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Event'
      replicaTimeout: 3600
      replicaRetryLimit: 2
      secrets: [
        { name: 'database-url', value: databaseUrl }
        { name: 'storage-connection', value: 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}' }
        { name: 'azure-openai-key', value: azureOpenAiKey }
      ]
      eventTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
        scale: {
          minExecutions: 0
          maxExecutions: 4
          pollingInterval: 30
          rules: [
            {
              name: 'queue-depth'
              type: 'azure-queue'
              metadata: {
                queueName: 'sift-ingest'
                queueLength: '16'
              }
              auth: [
                { secretRef: 'storage-connection', triggerParameter: 'connection' }
              ]
            }
          ]
        }
      }
    }
    template: {
      containers: [
        {
          name: 'ingest'
          image: apiImage
          command: ['python', '-m', 'apps.worker.consume']
          resources: {
            // Higher than the API: chunking and embedding a large RFC holds more than
            // serving a query does.
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'SIFT_DATABASE_URL', secretRef: 'database-url' }
            { name: 'SIFT_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection' }
            { name: 'SIFT_AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
            { name: 'SIFT_AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-key' }
            { name: 'SIFT_EMBEDDING_PROVIDER', value: empty(azureOpenAiKey) ? 'local' : 'azure_openai' }
          ]
        }
      ]
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output webUrl string = 'https://${web.properties.configuration.ingress.fqdn}'
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output storageAccount string = storage.name
output appInsightsName string = appInsights.name
