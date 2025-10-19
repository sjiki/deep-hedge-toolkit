# Monitoring Configuration

This directory contains configuration files for monitoring and observability.

## Grafana Agent Configuration

The `grafana-agent.yaml` file contains the configuration for Grafana Agent to collect metrics from the Deep Hedge Toolkit services and send them to Grafana Cloud.

### Overview

The configuration scrapes metrics from:
- **API Service** (`koyeb_api`): The main API endpoint on Koyeb
- **Worker Service** (`koyeb_worker`): Background worker processes on Koyeb
- **Redis** (`upstash_redis`): Upstash Redis instance metrics

All metrics are forwarded to Grafana Cloud for visualization and alerting.

### Configuration

Before deploying, update the following placeholders in `grafana-agent.yaml`:

1. **API Target**: Replace `<api>.koyeb.app` with your actual Koyeb API service URL
2. **Worker Target**: Replace `<worker>.koyeb.app` with your actual Koyeb worker service URL
3. **Redis Target**: Replace `<upstash-prom-endpoint-host>` with your Upstash Prometheus endpoint

### Environment Variables

Set the following environment variables for authentication with Grafana Cloud:

```bash
export GRAFANA_PROM_URL="https://prometheus-prod-xx.grafana.net/api/prom/push"
export GRAFANA_PROM_USER="<your-instance-id>"
export GRAFANA_PROM_APIKEY="<your-api-key>"
```

### Deployment

To run the Grafana Agent with this configuration:

```bash
grafana-agent --config.file=config/grafana-agent.yaml
```

### Metrics Collection

- **Scrape Interval**: 15 seconds (metrics are collected every 15 seconds)
- **Scrape Timeout**: 10 seconds (each scrape request times out after 10 seconds)
- **Metrics Path**: `/metrics` (standard Prometheus metrics endpoint)

### WAL (Write-Ahead Log)

The agent uses a Write-Ahead Log at `/tmp/wal` to ensure metrics are not lost if the remote write endpoint is temporarily unavailable.

### Troubleshooting

- Ensure your services expose Prometheus metrics at the `/metrics` endpoint
- Verify network connectivity to Grafana Cloud
- Check that environment variables are properly set
- Review agent logs (log level is set to `info`)

For more information, see the [Grafana Agent documentation](https://grafana.com/docs/agent/).
