targetScope = 'resourceGroup'

@description('Azure region for the Container Apps resources.')
param location string = resourceGroup().location

@description('Name of the Azure Container App.')
param containerAppName string = 'faceverification-api'

@description('Name of the Container Apps managed environment.')
param containerAppsEnvironmentName string = 'cae-faceverification-dev'

@description('Name of the Log Analytics workspace used by Container Apps.')
param logAnalyticsWorkspaceName string = 'law-faceverification-dev'

@description('Tags applied to all resources created by this deployment.')
param tags object = {
  app: 'faceverification'
  environment: 'dev'
  managedBy: 'bicep'
}

@description('Container image to deploy.')
param image string

@description('FastAPI container port.')
param targetPort int = 8000

@description('CPU cores allocated to the app container.')
param cpu string = '2.0'

@description('Memory allocated to the app container.')
param memory string = '4Gi'

@description('Demo username for the protected API endpoints.')
@secure()
param demoUsername string

@description('Demo password for the protected API endpoints.')
@secure()
param demoPassword string

@description('JWT signing secret for the FastAPI demo auth.')
@secure()
param jwtSecretKey string

@description('Optional GHCR username for private package pulls.')
param registryUsername string = ''

@description('Optional GHCR token for private package pulls. Leave empty for public packages.')
@secure()
param registryPassword string = ''

var registryServer = 'ghcr.io'
var appSecrets = concat([
  {
    name: 'demo-username'
    value: demoUsername
  }
  {
    name: 'demo-password'
    value: demoPassword
  }
  {
    name: 'jwt-secret-key'
    value: jwtSecretKey
  }
], empty(registryPassword) ? [] : [
  {
    name: 'ghcr-password'
    value: registryPassword
  }
])

var registries = empty(registryPassword) ? [] : [
  {
    server: registryServer
    username: registryUsername
    passwordSecretRef: 'ghcr-password'
  }
]

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvironmentName
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

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
      }
      secrets: appSecrets
      registries: registries
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          env: [
            {
              name: 'FACEVERIFICATION_DEVICE'
              value: 'cpu'
            }
            {
              name: 'FACEVERIFICATION_DEMO_USERNAME'
              secretRef: 'demo-username'
            }
            {
              name: 'FACEVERIFICATION_DEMO_PASSWORD'
              secretRef: 'demo-password'
            }
            {
              name: 'FACEVERIFICATION_JWT_SECRET_KEY'
              secretRef: 'jwt-secret-key'
            }
          ]
          resources: {
            cpu: json(cpu)
            memory: memory
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
