#!/bin/bash

################################################################################
# S3 CSI Driver 自动安装和配置脚本
# 
# 功能：
# 1. 自动创建 S3 CSI Driver 所需的 ServiceAccount 和 IAM 角色
# 2. 自动安装 S3 CSI Driver（Controller 和 Node DaemonSet）
# 3. 验证安装和配置是否正确
# 
# 使用场景：
# - 从零开始安装 S3 CSI Driver
# - 修复缺失的 s3-csi-driver-sa ServiceAccount
# - 为已存在的 S3 CSI Driver DaemonSet 添加 ServiceAccount
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 显示使用说明
usage() {
    cat << EOF
使用方法: $0 [选项]

必需参数:
    -c, --cluster-name      EKS 集群名称
    -r, --region           AWS 区域 (例如: us-east-1)
    -b, --bucket-name      S3 存储桶名称

可选参数:
    -p, --policy-name      IAM 策略名称 (默认: AmazonS3CSIDriverPolicy)
    -n, --role-name        IAM 角色名称 (默认: AmazonEKS_S3_CSI_DriverRole)
    --skip-role-creation   跳过 IAM 角色创建（如果角色已存在）
    -h, --help            显示此帮助信息

示例:
    # 创建 ServiceAccount 和 IAM 角色
    $0 -c my-cluster -r us-east-1 -b my-s3-bucket
    
    # 只创建 ServiceAccount（假设 IAM 角色已存在）
    $0 -c my-cluster -r us-east-1 -b my-bucket --skip-role-creation

说明:
    此脚本会：
    1. 检查并创建 IAM 策略（如果不存在）
    2. 检查 OIDC provider 是否存在
    3. 创建或更新 IAM 角色
    4. 创建 Kubernetes ServiceAccount 并关联 IAM 角色
    5. 安装 S3 CSI Driver（如果未安装）
       - 优先使用 Helm 安装
       - 如果 Helm 不可用，使用 kubectl 安装
    6. 验证配置是否正确

EOF
    exit 1
}

# 检查必要的命令是否存在
check_prerequisites() {
    log_info "检查必要工具..."
    
    local tools=("aws" "kubectl" "jq" "eksctl")
    local missing_tools=()
    local optional_missing=()
    
    # 检查必需工具
    for tool in "aws" "kubectl" "jq"; do
        if ! command -v $tool &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # 检查可选工具 (eksctl, helm)
    if ! command -v eksctl &> /dev/null; then
        optional_missing+=("eksctl")
    fi
    
    if ! command -v helm &> /dev/null; then
        optional_missing+=("helm")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "以下必需工具未安装: ${missing_tools[*]}"
        log_info "请先安装这些工具："
        for tool in "${missing_tools[@]}"; do
            case $tool in
                aws)
                    echo "  - AWS CLI: https://aws.amazon.com/cli/"
                    ;;
                kubectl)
                    echo "  - kubectl: https://kubernetes.io/docs/tasks/tools/"
                    ;;
                jq)
                    echo "  - jq: https://stedolan.github.io/jq/download/"
                    ;;
            esac
        done
        exit 1
    fi
    
    if [ ${#optional_missing[@]} -ne 0 ]; then
        log_warn "以下可选工具未安装: ${optional_missing[*]}"
        log_warn "如果需要自动创建 OIDC Provider，需要安装 eksctl"
        log_warn "如果需要通过 Helm 安装 S3 CSI Driver，需要安装 helm"
        log_info "安装方法："
        echo "  # macOS"
        echo "  brew install eksctl helm"
        echo ""
        echo "  # Linux"
        echo "  curl --silent --location \"https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_\$(uname -s)_amd64.tar.gz\" | tar xz -C /tmp"
        echo "  sudo mv /tmp/eksctl /usr/local/bin"
        echo "  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
        echo ""
    fi
    
    log_info "✓ 所有必要工具已安装"
}

# 解析命令行参数
parse_args() {
    CLUSTER_NAME=""
    REGION=""
    BUCKET_NAME=""
    POLICY_NAME="AmazonS3CSIDriverPolicy"
    ROLE_NAME="AmazonEKS_S3_CSI_DriverRole"
    SKIP_ROLE_CREATION="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -b|--bucket-name)
                BUCKET_NAME="$2"
                shift 2
                ;;
            -p|--policy-name)
                POLICY_NAME="$2"
                shift 2
                ;;
            -n|--role-name)
                ROLE_NAME="$2"
                shift 2
                ;;
            --skip-role-creation)
                SKIP_ROLE_CREATION="true"
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "未知参数: $1"
                usage
                ;;
        esac
    done
    
    # 验证必需参数
    if [[ -z "$CLUSTER_NAME" || -z "$REGION" || -z "$BUCKET_NAME" ]]; then
        log_error "缺少必需参数"
        usage
    fi
}

# 验证集群连接
verify_cluster_connection() {
    log_step "验证 Kubernetes 集群连接..."
    
    if ! kubectl cluster-info &>/dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        log_info "请确保已配置正确的 kubeconfig"
        exit 1
    fi
    
    # 验证集群名称是否匹配
    CURRENT_CLUSTER=$(kubectl config current-context)
    if [[ ! "$CURRENT_CLUSTER" =~ "$CLUSTER_NAME" ]]; then
        log_warn "当前 kubectl 上下文: $CURRENT_CLUSTER"
        log_warn "指定的集群名称: $CLUSTER_NAME"
        echo ""
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "操作已取消"
            exit 1
        fi
    fi
    
    log_info "✓ 集群连接正常"
}

# 获取 AWS 账户 ID
get_account_id() {
    log_step "获取 AWS 账户信息..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [[ -z "$ACCOUNT_ID" ]]; then
        log_error "无法获取 AWS 账户 ID"
        exit 1
    fi
    log_info "✓ AWS 账户 ID: $ACCOUNT_ID"
}

# 创建 IAM 策略
create_iam_policy() {
    log_step "检查并创建 IAM 策略..."
    
    # 创建策略 JSON 文件
    cat > /tmp/s3-csi-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MountpointFullBucketAccess",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET_NAME}"
            ]
        },
        {
            "Sid": "MountpointFullObjectAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:AbortMultipartUpload",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET_NAME}/*"
            ]
        }
    ]
}
EOF
    
    # 检查策略是否已存在
    POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn" --output text 2>/dev/null || echo "")
    
    if [[ -z "$POLICY_ARN" ]]; then
        log_info "创建 IAM 策略: ${POLICY_NAME}"
        POLICY_ARN=$(aws iam create-policy \
            --policy-name ${POLICY_NAME} \
            --policy-document file:///tmp/s3-csi-policy.json \
            --query 'Policy.Arn' \
            --output text)
        log_info "✓ IAM 策略创建成功: ${POLICY_ARN}"
    else
        log_info "✓ IAM 策略已存在: ${POLICY_ARN}"
        
        # 可选：更新策略
        log_info "更新策略版本..."
        VERSIONS=$(aws iam list-policy-versions --policy-arn ${POLICY_ARN} --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text)
        for version in $VERSIONS; do
            aws iam delete-policy-version --policy-arn ${POLICY_ARN} --version-id ${version} 2>/dev/null || true
        done
        aws iam create-policy-version \
            --policy-arn ${POLICY_ARN} \
            --policy-document file:///tmp/s3-csi-policy.json \
            --set-as-default > /dev/null
        log_info "✓ IAM 策略更新成功"
    fi
    
    # 清理临时文件
    rm -f /tmp/s3-csi-policy.json
}

# 检查并创建 OIDC provider
check_oidc_provider() {
    log_step "检查 OIDC Provider..."
    
    OIDC_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} --region ${REGION} --query "cluster.identity.oidc.issuer" --output text | cut -d '/' -f 5)
    
    if [[ -z "$OIDC_ID" ]]; then
        log_error "无法获取 OIDC Provider ID"
        log_error "请确认集群名称和区域是否正确"
        exit 1
    fi
    
    if aws iam list-open-id-connect-providers | grep -q ${OIDC_ID}; then
        log_info "✓ OIDC Provider 已存在: ${OIDC_ID}"
    else
        log_warn "OIDC Provider 不存在，正在自动创建..."
        
        # 检查是否安装了 eksctl
        if ! command -v eksctl &> /dev/null; then
            log_error "需要 eksctl 来创建 OIDC Provider"
            log_info "安装方法："
            echo "  # macOS"
            echo "  brew install eksctl"
            echo ""
            echo "  # Linux"
            echo "  curl --silent --location \"https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_\$(uname -s)_amd64.tar.gz\" | tar xz -C /tmp"
            echo "  sudo mv /tmp/eksctl /usr/local/bin"
            echo ""
            log_info "或者手动运行："
            echo "  eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} --region ${REGION} --approve"
            exit 1
        fi
        
        log_info "正在为集群 ${CLUSTER_NAME} 创建 OIDC Provider..."
        if eksctl utils associate-iam-oidc-provider \
            --cluster ${CLUSTER_NAME} \
            --region ${REGION} \
            --approve; then
            log_info "✓ OIDC Provider 创建成功"
        else
            log_error "OIDC Provider 创建失败"
            log_info "请手动运行以下命令："
            echo "  eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} --region ${REGION} --approve"
            exit 1
        fi
        
        # 验证创建结果
        sleep 3
        if aws iam list-open-id-connect-providers | grep -q ${OIDC_ID}; then
            log_info "✓ OIDC Provider 验证通过"
        else
            log_error "OIDC Provider 创建后验证失败"
            exit 1
        fi
    fi
    
    OIDC_PROVIDER="oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}"
    log_info "OIDC Provider URL: ${OIDC_PROVIDER}"
}

# 创建 IAM 角色
create_iam_role() {
    if [[ "$SKIP_ROLE_CREATION" == "true" ]]; then
        log_step "跳过 IAM 角色创建（使用现有角色）"
        ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
        
        # 验证角色是否存在
        if ! aws iam get-role --role-name ${ROLE_NAME} &>/dev/null; then
            log_error "IAM 角色 ${ROLE_NAME} 不存在"
            log_info "请移除 --skip-role-creation 参数以创建角色"
            exit 1
        fi
        log_info "✓ 使用现有 IAM 角色: ${ROLE_ARN}"
        return
    fi
    
    log_step "创建或更新 IAM 角色..."
    
    # 创建信任策略
    cat > /tmp/trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "${OIDC_PROVIDER}:aud": "sts.amazonaws.com",
                    "${OIDC_PROVIDER}:sub": "system:serviceaccount:kube-system:s3-csi-driver-sa"
                }
            }
        }
    ]
}
EOF
    
    # 检查角色是否已存在
    if aws iam get-role --role-name ${ROLE_NAME} &>/dev/null; then
        log_info "IAM 角色已存在，更新信任策略..."
        aws iam update-assume-role-policy \
            --role-name ${ROLE_NAME} \
            --policy-document file:///tmp/trust-policy.json
        log_info "✓ IAM 角色信任策略更新成功"
    else
        log_info "创建 IAM 角色: ${ROLE_NAME}"
        aws iam create-role \
            --role-name ${ROLE_NAME} \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --description "IAM role for S3 CSI Driver" > /dev/null
        log_info "✓ IAM 角色创建成功"
    fi
    
    # 附加策略到角色
    log_info "附加策略到角色..."
    aws iam attach-role-policy \
        --role-name ${ROLE_NAME} \
        --policy-arn ${POLICY_ARN} 2>/dev/null || log_info "策略已附加"
    
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
    log_info "✓ IAM 角色 ARN: ${ROLE_ARN}"
    
    # 清理临时文件
    rm -f /tmp/trust-policy.json
}

# 创建 Kubernetes ServiceAccount
create_service_account() {
    log_step "创建 Kubernetes ServiceAccount..."
    
    # 检查 ServiceAccount 是否已存在
    if kubectl get sa s3-csi-driver-sa -n kube-system &>/dev/null; then
        log_warn "ServiceAccount s3-csi-driver-sa 已存在"
        
        # 检查角色注解
        CURRENT_ROLE=$(kubectl get sa s3-csi-driver-sa -n kube-system -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")
        
        if [[ "$CURRENT_ROLE" == "$ROLE_ARN" ]]; then
            log_info "✓ ServiceAccount 已正确配置"
            return
        else
            log_warn "ServiceAccount 的角色注解不匹配"
            log_info "当前角色: ${CURRENT_ROLE:-<未设置>}"
            log_info "期望角色: ${ROLE_ARN}"
            
            read -p "是否更新 ServiceAccount? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_warn "跳过 ServiceAccount 更新"
                return
            fi
            
            log_info "更新 ServiceAccount 注解..."
            kubectl annotate sa s3-csi-driver-sa -n kube-system \
                eks.amazonaws.com/role-arn=${ROLE_ARN} \
                --overwrite
            log_info "✓ ServiceAccount 注解更新成功"
        fi
    else
        log_info "创建 ServiceAccount..."
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: s3-csi-driver-sa
  namespace: kube-system
  annotations:
    eks.amazonaws.com/role-arn: ${ROLE_ARN}
EOF
        log_info "✓ ServiceAccount 创建成功"
    fi
}

# 创建 RBAC 权限
create_rbac_permissions() {
    log_step "配置 RBAC 权限..."
    
    log_info "创建 ClusterRole 用于 S3 CSI Driver..."
    cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: s3-csi-driver-role
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
- apiGroups: [""]
  resources: ["persistentvolumes"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["volumeattachments"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apiextensions.k8s.io"]
  resources: ["customresourcedefinitions"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["s3.csi.aws.com"]
  resources: ["mountpoints3podattachments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
EOF
    
    log_info "创建 ClusterRoleBinding..."
    cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: s3-csi-driver-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: s3-csi-driver-role
subjects:
- kind: ServiceAccount
  name: s3-csi-driver-sa
  namespace: kube-system
EOF
    
    log_info "✓ RBAC 权限配置完成"
    
    # 验证 RBAC 配置
    log_info "验证 RBAC 配置..."
    if kubectl get clusterrole s3-csi-driver-role &>/dev/null && \
       kubectl get clusterrolebinding s3-csi-driver-binding &>/dev/null; then
        log_info "✓ RBAC 资源创建成功"
    else
        log_error "RBAC 资源创建验证失败"
        return 1
    fi
}

# 检查是否通过 EKS Addon 安装了 S3 CSI Driver
check_eks_addon() {
    local addon_status
    addon_status=$(aws eks describe-addon \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name aws-mountpoint-s3-csi-driver \
        --region "${REGION}" \
        --query 'addon.status' \
        --output text 2>/dev/null || echo "NOT_FOUND")
    echo "$addon_status"
}

# 安装 S3 CSI Driver
install_s3_csi_driver() {
    log_step "检查并安装 S3 CSI Driver..."

    # 检查是否已通过 EKS Addon 安装
    local addon_status
    addon_status=$(check_eks_addon)

    if [[ "$addon_status" != "NOT_FOUND" ]]; then
        log_info "✓ S3 CSI Driver 已通过 EKS Addon 安装 (状态: ${addon_status})"

        if [[ "$addon_status" == "ACTIVE" ]]; then
            log_info "EKS Addon 运行正常，跳过安装步骤"

            # 更新 Addon 的 ServiceAccount 配置
            log_info "更新 EKS Addon ServiceAccount 配置..."
            aws eks update-addon \
                --cluster-name "${CLUSTER_NAME}" \
                --addon-name aws-mountpoint-s3-csi-driver \
                --service-account-role-arn "${ROLE_ARN}" \
                --region "${REGION}" 2>/dev/null || log_warn "Addon ServiceAccount 更新失败，可能需要手动配置"

            # 显示组件状态
            if kubectl get daemonset s3-csi-node -n kube-system &>/dev/null; then
                NODE_READY=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.numberReady}' 2>/dev/null || echo "0")
                NODE_DESIRED=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo "0")
                log_info "  Node DaemonSet: ${NODE_READY}/${NODE_DESIRED} Ready"
            fi
            if kubectl get deployment s3-csi-controller -n kube-system &>/dev/null; then
                CONTROLLER_READY=$(kubectl get deployment s3-csi-controller -n kube-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
                CONTROLLER_DESIRED=$(kubectl get deployment s3-csi-controller -n kube-system -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
                log_info "  Controller: ${CONTROLLER_READY}/${CONTROLLER_DESIRED} Ready"
            fi
            return 0
        else
            log_warn "EKS Addon 状态异常: ${addon_status}"
            log_info "尝试删除并重新通过 EKS Addon 安装..."
            aws eks delete-addon \
                --cluster-name "${CLUSTER_NAME}" \
                --addon-name aws-mountpoint-s3-csi-driver \
                --region "${REGION}" 2>/dev/null || true
            log_info "等待 Addon 删除完成..."
            sleep 30
        fi
    fi

    # 检查是否存在由 EKS 管理但非 Addon API 注册的残留资源
    if kubectl get daemonset s3-csi-node -n kube-system &>/dev/null; then
        local managed_by
        managed_by=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.metadata.labels.app\.kubernetes\.io/managed-by}' 2>/dev/null || echo "")
        if [[ "$managed_by" == "EKS" ]]; then
            log_warn "检测到 EKS 管理的 S3 CSI Driver 资源（非 Addon API），尝试通过 EKS Addon API 安装..."
            if aws eks create-addon \
                --cluster-name "${CLUSTER_NAME}" \
                --addon-name aws-mountpoint-s3-csi-driver \
                --service-account-role-arn "${ROLE_ARN}" \
                --region "${REGION}" \
                --resolve-conflicts OVERWRITE 2>/dev/null; then
                log_info "✓ EKS Addon 创建成功，等待就绪..."
                wait_for_addon_ready
                return $?
            else
                log_warn "EKS Addon API 安装失败，将直接使用已有资源"
                log_info "更新 DaemonSet ServiceAccount..."
                kubectl patch daemonset s3-csi-node -n kube-system \
                    -p '{"spec":{"template":{"spec":{"serviceAccountName":"s3-csi-driver-sa"}}}}' 2>/dev/null || true
                return 0
            fi
        fi
    fi

    # 检查是否已安装且健康（非 EKS 管理）
    if kubectl get daemonset s3-csi-node -n kube-system &>/dev/null; then
        log_info "✓ S3 CSI Driver 已安装"
        NODE_READY=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.numberReady}' 2>/dev/null || echo "0")
        NODE_DESIRED=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo "0")
        log_info "  Node: ${NODE_READY}/${NODE_DESIRED} Ready"

        if kubectl get deployment s3-csi-controller -n kube-system &>/dev/null; then
            CONTROLLER_READY=$(kubectl get deployment s3-csi-controller -n kube-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            CONTROLLER_DESIRED=$(kubectl get deployment s3-csi-controller -n kube-system -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
            log_info "  Controller: ${CONTROLLER_READY}/${CONTROLLER_DESIRED} Ready"
        fi

        if [[ "$NODE_READY" -gt 0 ]]; then
            log_info "✓ S3 CSI Driver 运行正常"
            return 0
        else
            log_warn "S3 CSI Driver 状态异常，尝试重新安装..."
        fi
    else
        log_info "S3 CSI Driver 未安装，开始安装..."
    fi

    # 优先尝试 EKS Addon 方式安装
    log_info "尝试通过 EKS Addon 安装 S3 CSI Driver..."
    if aws eks create-addon \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name aws-mountpoint-s3-csi-driver \
        --service-account-role-arn "${ROLE_ARN}" \
        --region "${REGION}" \
        --resolve-conflicts OVERWRITE 2>/dev/null; then
        log_info "✓ EKS Addon 创建成功，等待就绪..."
        wait_for_addon_ready
        return $?
    else
        log_warn "EKS Addon 安装失败，回退到 Helm 安装..."
    fi

    # 回退: 使用 Helm 安装
    if command -v helm &> /dev/null; then
        log_info "使用 Helm 安装 S3 CSI Driver..."

        log_info "添加 Helm repository..."
        helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver 2>/dev/null || true
        helm repo update

        log_info "安装 S3 CSI Driver..."
        if helm upgrade --install aws-mountpoint-s3-csi-driver \
            aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver \
            --namespace kube-system \
            --set node.serviceAccount.name=s3-csi-driver-sa \
            --set node.serviceAccount.create=false; then
            log_info "✓ S3 CSI Driver 安装成功（使用 Helm）"
        else
            log_error "Helm 安装失败，尝试使用 kubectl 方式..."
            install_s3_csi_driver_kubectl
            return $?
        fi
    else
        log_warn "未检测到 Helm，使用 kubectl 安装 S3 CSI Driver..."
        install_s3_csi_driver_kubectl
        return $?
    fi

    wait_for_driver_ready
}

# 等待 EKS Addon 就绪
wait_for_addon_ready() {
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        local status
        status=$(check_eks_addon)

        if [[ "$status" == "ACTIVE" ]]; then
            log_info "✓ EKS Addon 已就绪 (ACTIVE)"
            return 0
        elif [[ "$status" == "CREATE_FAILED" || "$status" == "DEGRADED" ]]; then
            log_error "EKS Addon 状态异常: ${status}"
            log_info "请检查: aws eks describe-addon --cluster-name ${CLUSTER_NAME} --addon-name aws-mountpoint-s3-csi-driver --region ${REGION}"
            return 1
        fi

        log_info "等待 EKS Addon 就绪... (${attempt}/${max_attempts}) 当前状态: ${status}"
        sleep 10
        ((attempt++))
    done

    log_error "EKS Addon 就绪超时"
    return 1
}

# 等待 Driver 组件就绪
wait_for_driver_ready() {
    log_info "等待 S3 CSI Driver 组件就绪..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        NODE_READY=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.numberReady}' 2>/dev/null || echo "0")

        if [[ "$NODE_READY" -gt 0 ]]; then
            log_info "✓ S3 CSI Driver 组件已就绪"
            echo ""
            kubectl get daemonset s3-csi-node -n kube-system
            kubectl get deployment s3-csi-controller -n kube-system 2>/dev/null || true
            return 0
        fi

        log_info "等待组件就绪... (${attempt}/${max_attempts})"
        sleep 10
        ((attempt++))
    done

    log_error "S3 CSI Driver 安装超时"
    log_info "请手动检查组件状态："
    echo "  kubectl get daemonset s3-csi-node -n kube-system"
    echo "  kubectl get deployment s3-csi-controller -n kube-system"
    echo "  kubectl logs -n kube-system -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver"
    return 1
}

# 使用 kubectl 安装 S3 CSI Driver
install_s3_csi_driver_kubectl() {
    log_info "使用 kubectl 安装 S3 CSI Driver..."

    local CSI_VERSION="v1.10.0"
    local BASE_URL="https://raw.githubusercontent.com/awslabs/mountpoint-s3-csi-driver/${CSI_VERSION}/deploy/kubernetes/base"

    log_info "下载 S3 CSI Driver manifest (版本: ${CSI_VERSION})..."

    local TEMP_DIR="/tmp/s3-csi-driver-$$"
    mkdir -p "${TEMP_DIR}"

    local FILES=("kustomization.yaml" "csidriver.yaml" "node.yaml" "node-windows.yaml" "controller.yaml" "clusterrole.yaml" "clusterrolebinding.yaml")
    local download_ok=true

    for f in "${FILES[@]}"; do
        if ! curl -sSL --fail "${BASE_URL}/${f}" -o "${TEMP_DIR}/${f}" 2>/dev/null; then
            log_warn "下载 ${f} 失败（可能不存在），跳过"
            rm -f "${TEMP_DIR}/${f}"
        fi
    done

    if [[ ! -f "${TEMP_DIR}/kustomization.yaml" ]]; then
        log_error "下载 kustomization.yaml 失败"
        rm -rf "${TEMP_DIR}"
        return 1
    fi

    if kubectl apply -k "${TEMP_DIR}"; then
        log_info "✓ S3 CSI Driver manifest 应用成功"

        log_info "更新 DaemonSet ServiceAccount..."
        kubectl patch daemonset s3-csi-node -n kube-system \
            -p '{"spec":{"template":{"spec":{"serviceAccountName":"s3-csi-driver-sa"}}}}' 2>/dev/null || true

        rm -rf "${TEMP_DIR}"
        return 0
    else
        log_error "应用 S3 CSI Driver manifest 失败"
        rm -rf "${TEMP_DIR}"
        return 1
    fi
}

# 验证配置
verify_configuration() {
    log_step "验证配置..."
    
    echo ""
    log_info "=== ServiceAccount 信息 ==="
    kubectl get sa s3-csi-driver-sa -n kube-system -o yaml | grep -A 5 "annotations:"
    
    echo ""
    log_info "=== IAM 角色信息 ==="
    echo "  角色名称: ${ROLE_NAME}"
    echo "  角色 ARN: ${ROLE_ARN}"
    echo "  策略 ARN: ${POLICY_ARN}"
    
    echo ""
    log_info "=== S3 CSI Driver 组件状态 ==="
    if kubectl get deployment s3-csi-controller -n kube-system &>/dev/null; then
        kubectl get deployment s3-csi-controller -n kube-system
    else
        log_warn "未找到 s3-csi-controller Deployment"
    fi
    
    echo ""
    if kubectl get daemonset s3-csi-node -n kube-system &>/dev/null; then
        kubectl get daemonset s3-csi-node -n kube-system
        
        echo ""
        log_info "检查 DaemonSet 事件..."
        kubectl describe daemonset s3-csi-node -n kube-system | grep -A 10 "Events:"
    else
        log_warn "未找到 s3-csi-node DaemonSet"
    fi
    
    echo ""
}

# 重启 DaemonSet
restart_daemonset() {
    log_step "重启 S3 CSI Driver DaemonSet..."
    
    if ! kubectl get daemonset s3-csi-node -n kube-system &>/dev/null; then
        log_warn "未找到 s3-csi-node DaemonSet，跳过重启"
        return 0
    fi
    
    # 检查 DaemonSet 是否有 Pods 运行
    CURRENT_PODS=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.currentNumberScheduled}' 2>/dev/null || echo "0")
    
    if [[ "$CURRENT_PODS" == "0" ]]; then
        log_warn "DaemonSet 当前没有运行的 Pods，触发重启以创建 Pods..."
        
        # 使用 rollout restart 触发重新创建
        if kubectl rollout restart daemonset s3-csi-node -n kube-system; then
            log_info "✓ DaemonSet 重启命令已执行"
            
            # 等待 rollout 完成
            log_info "等待 DaemonSet rollout 完成..."
            sleep 5
        else
            log_error "DaemonSet 重启失败"
            return 1
        fi
    else
        log_info "✓ DaemonSet 已有 ${CURRENT_PODS} 个 Pods 运行，无需重启"
        
        # 检查是否有失败的 Pod 创建事件
        FAILED_EVENTS=$(kubectl describe daemonset s3-csi-node -n kube-system 2>/dev/null | grep -c "FailedCreate" || echo "0")
        
        if [[ "$FAILED_EVENTS" -gt 0 ]]; then
            log_warn "检测到之前的 Pod 创建失败事件"
            read -p "是否重启 DaemonSet 以清除错误状态? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "重启 DaemonSet..."
                kubectl rollout restart daemonset s3-csi-node -n kube-system
                log_info "✓ DaemonSet 已重启"
                sleep 5
            fi
        fi
    fi
    
    echo ""
}

# 等待 Pods 启动
wait_for_pods() {
    log_step "等待 S3 CSI Driver Pods 启动..."
    
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        READY_PODS=$(kubectl get pods -n kube-system -l app=s3-csi-node --field-selector=status.phase=Running 2>/dev/null | grep -c "Running" || echo "0")
        DESIRED_PODS=$(kubectl get daemonset s3-csi-node -n kube-system -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo "0")
        
        if [[ "$READY_PODS" -gt 0 ]]; then
            log_info "✓ S3 CSI Driver Pods 正在运行: ${READY_PODS}/${DESIRED_PODS}"
            
            echo ""
            log_info "=== Pods 状态 ==="
            kubectl get pods -n kube-system -l app=s3-csi-node -o wide
            return 0
        fi
        
        log_info "等待 Pods 启动... (${attempt}/${max_attempts}) - 当前: ${READY_PODS}/${DESIRED_PODS}"
        sleep 10
        ((attempt++))
    done
    
    log_warn "Pods 启动超时"
    log_info "请检查 DaemonSet 和 Pod 状态："
    echo "  kubectl get daemonset s3-csi-node -n kube-system"
    echo "  kubectl get pods -n kube-system -l app=s3-csi-node"
    echo "  kubectl describe daemonset s3-csi-node -n kube-system"
    return 1
}

# 打印后续步骤
print_next_steps() {
    echo ""
    log_info "=========================================="
    log_info "✓ S3 CSI Driver 安装和配置完成"
    log_info "=========================================="
    echo ""
    log_info "后续步骤："
    echo ""
    echo "  1. 检查 S3 CSI Driver 组件状态："
    echo "     kubectl get deployment s3-csi-controller -n kube-system"
    echo "     kubectl get daemonset s3-csi-node -n kube-system"
    echo "     kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver"
    echo ""
    echo "  2. 如果组件未正常运行，检查详细信息："
    echo "     kubectl describe deployment s3-csi-controller -n kube-system"
    echo "     kubectl describe daemonset s3-csi-node -n kube-system"
    echo "     kubectl logs -n kube-system -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver"
    echo ""
    echo "  3. 验证 CSI Driver 是否注册成功："
    echo "     kubectl get csidrivers"
    echo "     # 应该看到 s3.csi.aws.com"
    echo ""
    echo "  4. 重新部署您的应用 Pod（如果之前部署失败）："
    echo "     kubectl delete pod <your-pod-name>"
    echo "     # 或者重启 Deployment"
    echo "     kubectl rollout restart deployment <your-deployment-name>"
    echo ""
    echo "  5. 验证 S3 卷挂载："
    echo "     kubectl describe pod <your-pod-name>"
    echo "     kubectl exec <your-pod-name> -- df -h"
    echo ""
    log_info "文档参考："
    echo "  官方文档: https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html"
    echo "  GitHub: https://github.com/awslabs/mountpoint-s3-csi-driver"
    log_info "=========================================="
}

# 主函数
main() {
    echo ""
    log_info "=========================================="
    log_info "S3 CSI Driver 自动安装和配置脚本"
    log_info "=========================================="
    echo ""
    
    parse_args "$@"
    check_prerequisites
    verify_cluster_connection
    get_account_id
    
    echo ""
    log_info "配置参数："
    echo "  集群名称: ${CLUSTER_NAME}"
    echo "  AWS 区域: ${REGION}"
    echo "  AWS 账户: ${ACCOUNT_ID}"
    echo "  S3 存储桶: ${BUCKET_NAME}"
    echo "  IAM 策略: ${POLICY_NAME}"
    echo "  IAM 角色: ${ROLE_NAME}"
    echo "  跳过角色创建: ${SKIP_ROLE_CREATION}"
    echo ""
    
    read -p "确认以上配置是否正确? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "操作已取消"
        exit 0
    fi
    
    echo ""
    create_iam_policy
    check_oidc_provider
    create_iam_role
    create_service_account
    create_rbac_permissions
    install_s3_csi_driver
    verify_configuration
    restart_daemonset
    wait_for_pods
    print_next_steps
    
    log_info "脚本执行完成!"
}

# 执行主函数
main "$@"

