# Triton Metrics 上报到 Amazon Managed Prometheus (AMP)

## 架构

```
Triton Pods (port 10087/metrics)
    ↓ (自动发现和采集)
AMP Managed Scraper
    ↓
AMP Workspace (Your-created-workspace-id)
    ↓
Grafana (可视化)
```

## 当前配置

- **集群**: eks-cluster-hypd-test-whisper
- **Region**: us-east-2
- **Workspace**: ws-4b3e005c-2c30-4bf8-a189-e0c6e53329f8
- **Scraper ID**: s-7e66b07a-e80d-41a2-8af5-dfdde4de4ec5
- **Triton Metrics Port**: 10087

## 部署步骤

### 前置条件

1. **EKS 集群必须启用 Private Access**:
```bash
aws eks update-cluster-config \
  --name eks-cluster-hypd-test-whisper \
  --region us-east-2 \
  --resources-vpc-config endpointPrivateAccess=true,endpointPublicAccess=true
```

2. **Triton Pods 必须有 Prometheus annotations**:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "10087"
  prometheus.io/path: "/metrics"
```

### 创建 AMP Scraper

**通过 EKS Console 创建** (推荐):

1. 打开 EKS Console: https://console.aws.amazon.com/eks/home?region=us-east-2
2. 选择集群: `eks-cluster-hypd-test-whisper`
3. 点击 **Observability** tab
4. 点击 **Add scraper**
5. 配置:
   - **Workspace**: `ws-4b3e005c-2c30-4bf8-a189-e0c6e53329f8`
   - **Configuration**: 上传 `triton-scraper-config.yaml`
6. 等待 10-15 分钟直到 scraper 状态变为 `ACTIVE`

### 验证 Metrics 采集

```bash
# 1. 检查 scraper 状态
aws amp describe-scraper \
  --scraper-id s-7e66b07a-e80d-41a2-8af5-dfdde4de4ec5 \
  --region us-east-2 \
  --query 'scraper.status.statusCode'

# 2. 安装 awscurl
pip3 install awscurl

# 3. 查询 Triton metrics
WORKSPACE_ID="ws-4b3e005c-2c30-4bf8-a189-e0c6e53329f8"
REGION="us-east-2"
ENDPOINT="https://aps-workspaces.${REGION}.amazonaws.com/workspaces/${WORKSPACE_ID}"

awscurl --service aps --region $REGION \
  "${ENDPOINT}/api/v1/query?query=nv_inference_request_success" | python3 -m json.tool
```

## Triton Metrics 说明

采集的主要 metrics:

- `nv_inference_request_success` - 成功的推理请求数
- `nv_inference_request_failure` - 失败的推理请求数
- `nv_inference_count` - 推理次数
- `nv_inference_exec_count` - 模型执行次数
- `nv_inference_request_duration_us` - 请求持续时间(微秒)
- `nv_inference_queue_duration_us` - 队列等待时间(微秒)
- `nv_inference_compute_infer_duration_us` - 推理计算时间(微秒)
- `nv_gpu_utilization` - GPU 利用率
- `nv_gpu_memory_used_bytes` - GPU 内存使用

## Grafana 配置

### 添加 AMP 数据源

1. 打开 Grafana
2. Configuration → Data Sources → Add data source
3. 选择 **Prometheus**
4. 配置:
   - **Name**: Amazon Managed Prometheus
   - **URL**: `https://aps-workspaces.us-east-2.amazonaws.com/workspaces/ws-4b3e005c-2c30-4bf8-a189-e0c6e53329f8`
   - **Auth**: 启用 **SigV4 auth**
   - **Default Region**: us-east-2
5. Save & Test

### 示例查询

```promql
# 请求成功率
rate(nv_inference_request_success[5m])

# 平均推理延迟 (毫秒)
rate(nv_inference_request_duration_us[5m]) / rate(nv_inference_count[5m]) / 1000

# GPU 利用率
nv_gpu_utilization

# 按 pod 分组的请求率
sum by (pod) (rate(nv_inference_request_success[5m]))

# 队列等待时间
rate(nv_inference_queue_duration_us[5m]) / rate(nv_inference_count[5m]) / 1000
```

## 故障排查

### 1. 检查 Triton metrics 端点

```bash
kubectl exec -it <triton-pod> -- curl http://localhost:10087/metrics
```

### 2. 检查 Pod annotations

```bash
kubectl get pods -l service=whisper-triton -o yaml | grep -A 5 annotations
```

### 3. 检查 Scraper 状态

```bash
aws amp describe-scraper --scraper-id s-7e66b07a-e80d-41a2-8af5-dfdde4de4ec5 --region us-east-2
```

### 4. 查看 up metric

```bash
# up=1: 采集正常
# up=0: 发现端点但无法采集
# 无 up metric: 无法发现端点
awscurl --service aps --region us-east-2 \
  "https://aps-workspaces.us-east-2.amazonaws.com/workspaces/ws-4b3e005c-2c30-4bf8-a189-e0c6e53329f8/api/v1/query?query=up" | python3 -m json.tool
```

## 管理 Scraper

### 列出所有 scrapers

```bash
aws amp list-scrapers --region us-east-2
```

### 删除 scraper

```bash
aws amp delete-scraper --scraper-id <scraper-id> --region us-east-2
```

### 更新 scraper 配置

```bash
# 需要先获取新配置的 base64 编码
CONFIG_BLOB=$(base64 -w 0 triton-scraper-config.yaml)

aws amp update-scraper \
  --scraper-id s-7e66b07a-e80d-41a2-8af5-dfdde4de4ec5 \
  --region us-east-2 \
  --scrape-configuration "{\"configurationBlob\":\"$CONFIG_BLOB\"}"
```

## 文件说明

- `triton-scraper-config.yaml` - Triton metrics 采集配置
- `triton-deployment-with-metrics.yaml` - 带 Prometheus annotations 的 Triton deployment 示例

## 限制

- 最小采集间隔: 30 秒
- 每个 scraper 最多 30,000 个端点
- 每个 /metrics 响应最大 50MB
- 每个 region 每个账户最多 10 个 scrapers
