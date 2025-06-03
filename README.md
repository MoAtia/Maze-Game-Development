# Maze-Game-Development

## Overview
This repository provides an API for real-time hand gesture classification, designed to be integrated into interactive applications such as games. The core of the system is a machine learning model served using FastAPI, enabling efficient and scalable inference.

## Features

- **Monitoring & Observability:**
  - **Prometheus Metrics Exposure:** The API exposes Prometheus-compatible metrics at `/metrics` for real-time monitoring of model, data, and server health.
  - **Grafana Dashboard:** A sample `dashboard.json` is included for visualizing key metrics in Grafana.
  - **Three Key Metrics:**
    - **Model Metric:** The letancy of prediction, to ensure the real-time performance and avoid any bottelneck by taking consedring the latency Vs model complexity tradoff.
    - **Data Metric:** Data drift monitoring by track the landmarks' coordinates distribution.
    - **Server Metric:** Request latency logging to ensure API performance, identify bottlenecks and to identify the proper rate for rendering the game.


## Monitoring & Observability
- **Prometheus Metrics:** Access at `http://localhost:8000/metrics` when running locally or via Docker Compose.
- **Grafana Dashboard:** Import the provided `dashboard.json` into Grafana to visualize model confidence, data drift, and latency metrics.
