#!/bin/bash

# 诊断并修复 AWS Load Balancer 问题
# 支持：
#   1. 检测是否安装了 AWS Load Balancer Controller
#   2. 修复 IAM 权限问题
#   3. 安装 AWS Load Balancer Controller（如需要）
#
# 使用方法:
#   ./create_lb_controller.sh -v vpc-xxxxxxxxx -c cluster-name
#
# 参数:
#   -v VPC_ID           VPC ID (必需)
#   -c CLUSTER_NAME     EKS 集群名称 (必需)
#   -r REGION           AWS 区域 (可选，默认从 AWS CLI 配置获取)
#   -h                  显示帮助信息

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查 AWS Load Balancer Controller 是否已安装
check_lb_controller() {
    echo -e "${BLUE}[检查]${NC} 检查 AWS Load Balancer Controller 状态..."
    
    # 检查 deployment 是否存在
    if kubectl get deployment aws-load-balancer-controller -n kube-system >/dev/null 2>&1; then
        echo -e "${GREEN}[信息]${NC} AWS Load Balancer Controller deployment 已存在"
        
        # 检查 pod 状态
        READY_PODS=$(kubectl get deployment aws-load-balancer-controller -n kube-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        DESIRED_PODS=$(kubectl get deployment aws-load-balancer-controller -n kube-system -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
        
        if [ "$READY_PODS" = "$DESIRED_PODS" ] && [ "$READY_PODS" != "0" ]; then
            echo -e "${GREEN}[成功]${NC} AWS Load Balancer Controller 运行正常 ($READY_PODS/$DESIRED_PODS pods ready)"
            return 0
        else
            echo -e "${YELLOW}[警告]${NC} AWS Load Balancer Controller pods 未就绪 ($READY_PODS/$DESIRED_PODS pods ready)"
            return 1
        fi
    else
        echo -e "${RED}[错误]${NC} AWS Load Balancer Controller 未安装"
        return 1
    fi
}

# 显示使用帮助
show_usage() {
    echo "使用方法: $0 -v <VPC_ID> -c <CLUSTER_NAME> [-r <REGION>]"
    echo ""
    echo "必需参数:"
    echo "  -v VPC_ID           VPC ID (格式: vpc-xxxxxxxxx)"
    echo "  -c CLUSTER_NAME     EKS/HyperPod 集群名称"
    echo ""
    echo "可选参数:"
    echo "  -r REGION           AWS 区域 (默认: 从 AWS CLI 配置获取)"
    echo "  -h                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -v vpc-070ef659074929797 -c sagemaker-mini-asr-d463da31-eks"
    echo "  $0 -v vpc-070ef659074929797 -c sagemaker-mini-asr-d463da31-eks -r us-east-1"
    echo ""
    echo "如何获取参数:"
    echo ""
    echo "1. 获取 VPC ID:"
    echo "   INSTANCE_ID=\$(kubectl get nodes -o jsonpath='{.items[0].spec.providerID}' | grep -o 'i-[a-f0-9]*')"
    echo "   aws ec2 describe-instances --instance-ids \$INSTANCE_ID --query 'Reservations[0].Instances[0].VpcId' --output text"
    echo ""
    echo "2. 获取集群名称:"
    echo "   kubectl config current-context | awk -F'/' '{print \$2}'"
    echo ""
}

# 全局变量
CONTROLLER_TYPE=""
NEEDS_INSTALL=false
VPC_ID=""
CLUSTER_NAME=""
AWS_REGION=""

# 解析命令行参数
while getopts "v:c:r:h" opt; do
    case $opt in
        v)
            VPC_ID="$OPTARG"
            ;;
        c)
            CLUSTER_NAME="$OPTARG"
            ;;
        r)
            AWS_REGION="$OPTARG"
            ;;
        h)
            show_usage
            exit 0
            ;;
        \?)
            echo -e "${RED}✗${NC} 无效的参数: -$OPTARG" >&2
            echo ""
            show_usage
            exit 1
            ;;
        :)
            echo -e "${RED}✗${NC} 参数 -$OPTARG 需要一个值" >&2
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$VPC_ID" ] || [ -z "$CLUSTER_NAME" ]; then
    echo -e "${RED}✗${NC} 错误: 缺少必需参数"
    echo ""
    if [ -z "$VPC_ID" ]; then
        echo "  缺少 VPC ID (-v 参数)"
    fi
    if [ -z "$CLUSTER_NAME" ]; then
        echo "  缺少集群名称 (-c 参数)"
    fi
    echo ""
    show_usage
    exit 1
fi

echo "=========================================="
echo "AWS Load Balancer 问题诊断和修复"
echo "=========================================="
echo ""
echo "此脚本将:"
echo "  1. 诊断当前 Load Balancer 配置"
echo "  2. 检测 AWS Load Balancer Controller 状态"
echo "  3. 修复 IAM 权限（如适用）"
echo "  4. 安装 Controller（如需要）"
echo ""

# 获取当前配置
echo "=========================================="
echo "阶段 1: 环境检测和配置"
echo "=========================================="
echo ""
echo "步骤 1.1: 验证输入参数..."

# 验证 VPC ID 格式
if [[ ! $VPC_ID =~ ^vpc-[a-f0-9]{8,17}$ ]]; then
    echo -e "${RED}✗${NC} VPC ID 格式不正确: $VPC_ID"
    echo "VPC ID 应该以 'vpc-' 开头，后跟 8-17 位十六进制字符"
    exit 1
fi

# 如果未指定区域，尝试自动获取
if [ -z "$AWS_REGION" ]; then
    # 方式 1: 从 AWS CLI 配置
    AWS_REGION=$(aws configure get region 2>/dev/null || echo "")
    # 方式 2: 从环境变量
    if [ -z "$AWS_REGION" ]; then
        AWS_REGION=${AWS_DEFAULT_REGION:-""}
    fi
    # 方式 3: 从 kubectl 上下文
    if [ -z "$AWS_REGION" ]; then
        AWS_REGION=$(kubectl config current-context 2>/dev/null | grep -o '[a-z]*-[a-z]*-[0-9]' | head -1 || echo "")
    fi
    # 如果还是无法获取，报错
    if [ -z "$AWS_REGION" ]; then
        echo -e "${RED}✗${NC} 无法自动检测 AWS 区域"
        echo "请使用 -r 参数指定区域，例如: -r us-east-1"
        exit 1
    fi
fi

# 获取账户 ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# 验证 VPC 是否存在
echo "验证 VPC ID..."
if ! aws ec2 describe-vpcs --region "$AWS_REGION" --vpc-ids "$VPC_ID" >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} VPC 不存在或无权限访问: $VPC_ID (区域: $AWS_REGION)"
    exit 1
fi

echo -e "${GREEN}✓${NC} AWS 账户 ID: $ACCOUNT_ID"
echo -e "${GREEN}✓${NC} AWS 区域: $AWS_REGION"
echo -e "${GREEN}✓${NC} 集群名称: $CLUSTER_NAME"
echo -e "${GREEN}✓${NC} VPC ID: $VPC_ID"
echo ""

# 检测 Load Balancer Controller 类型
echo "步骤 1.2: 检测 Load Balancer Controller..."

# 检查标准的 AWS Load Balancer Controller
if kubectl get deployment -n kube-system aws-load-balancer-controller >/dev/null 2>&1; then
    CONTROLLER_TYPE="aws-lb-controller"
    echo -e "${GREEN}✓${NC} 发现标准 AWS Load Balancer Controller"
    kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
    NEEDS_INSTALL=false
# 检查 HyperPod 自带的 ALB Controller
elif kubectl get deployment -n kube-system hyperpod-inference-operator-alb >/dev/null 2>&1; then
    CONTROLLER_TYPE="hyperpod-alb"
    echo -e "${YELLOW}!${NC} 发现 HyperPod Inference Operator ALB"
    kubectl get pods -n kube-system -l app.kubernetes.io/name=alb
    echo ""
    echo -e "${BLUE}ℹ${NC}  注意: HyperPod 自带的 operator 可能不完全支持标准 NLB annotations"
    NEEDS_INSTALL=false
else
    CONTROLLER_TYPE="none"
    echo -e "${RED}✗${NC} 未发现任何 Load Balancer Controller"
    NEEDS_INSTALL=true
fi
echo ""

# 根据检测结果决定操作类型
if [ "$NEEDS_INSTALL" = true ]; then
    echo -e "${YELLOW}▶${NC} 未安装 AWS Load Balancer Controller"
    echo "将进行安装..."
    INSTALL_CONTROLLER=true
else
    echo -e "${GREEN}✓${NC} 已安装 Controller: $CONTROLLER_TYPE"
    
    if [ "$CONTROLLER_TYPE" == "hyperpod-alb" ]; then
        echo -e "${BLUE}ℹ${NC}  将修复 HyperPod ALB Operator 的 IAM 权限..."
        # HyperPod 使用硬编码的 Role 名称（需要从现有环境获取）
        SA_ROLE=$(kubectl get sa -n kube-system -o jsonpath='{.items[?(@.metadata.name=="hyperpod-inference-operator-alb")].metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")
        if [ -n "$SA_ROLE" ]; then
            ROLE_NAME=$(echo $SA_ROLE | awk -F'/' '{print $NF}')
            echo "检测到 Role: $ROLE_NAME"
        else
            echo -e "${YELLOW}!${NC} 无法自动检测 Role，请检查 HyperPod operator 配置"
            exit 1
        fi
    else
        # 标准 AWS LB Controller - 查找 ServiceAccount 对应的 Role
        SA_ROLE=$(kubectl get sa aws-load-balancer-controller -n kube-system -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")
        if [ -n "$SA_ROLE" ]; then
            ROLE_NAME=$(echo $SA_ROLE | awk -F'/' '{print $NF}')
            echo "检测到 Role: $ROLE_NAME"
        else
            echo -e "${YELLOW}!${NC} 无法自动检测 Role，将使用默认值"
            ROLE_NAME="AmazonEKSLoadBalancerControllerRole"
        fi
    fi
    INSTALL_CONTROLLER=false
fi
echo ""

# 检测集群类型（EKS 或 HyperPod）
detect_cluster_type() {
    echo "检测集群类型..." >&2
    
    # 尝试通过 EKS API 查询
    if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 检测到标准 EKS 集群" >&2
        echo "STANDARD_EKS"
        return 0
    else
        # 检查是否是 HyperPod（从 context 判断）
        CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
        if echo "$CURRENT_CONTEXT" | grep -q "sagemaker"; then
            echo -e "${YELLOW}!${NC} 检测到 SageMaker HyperPod 集群" >&2
            echo -e "${BLUE}ℹ${NC}  HyperPod 集群不会出现在 EKS API 中" >&2
            echo "HYPERPOD"
            return 0
        else
            echo -e "${YELLOW}!${NC} 无法通过 EKS API 查询集群" >&2
            echo -e "${BLUE}ℹ${NC}  将按 HyperPod 模式处理" >&2
            echo "HYPERPOD"
            return 0
        fi
    fi
}

# 验证 Role 是否存在
echo "=========================================="
echo "阶段 2: IAM 权限配置"
echo "=========================================="
echo ""
echo "步骤 2.1: 验证 IAM Role..."
if [ -n "$ROLE_NAME" ]; then
    echo "Role 名称: $ROLE_NAME"
fi

ROLE_EXISTS=false
if aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} IAM Role 存在"
    ROLE_EXISTS=true
else
    echo -e "${YELLOW}!${NC} IAM Role 不存在: $ROLE_NAME"
    
    if [ "$INSTALL_CONTROLLER" = true ]; then
        echo "    将在安装过程中创建该 Role..."
    else
        echo "    将自动创建该 Role..."
    fi
    ROLE_EXISTS=false
fi
echo ""

# 如果需要安装 Controller
if [ "$INSTALL_CONTROLLER" = true ]; then
    echo "=========================================="
    echo "阶段 3: 安装 AWS Load Balancer Controller"
    echo "=========================================="
    echo ""
    
    # 检查必要工具
    echo "步骤 3.1: 检查必要工具..."
    MISSING_TOOLS=()
    for cmd in eksctl helm jq; do
        if ! command -v $cmd &> /dev/null; then
            MISSING_TOOLS+=($cmd)
        fi
    done
    
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        echo -e "${RED}✗${NC} 缺少必要工具: ${MISSING_TOOLS[*]}"
        echo ""
        echo "请安装缺少的工具:"
        for tool in "${MISSING_TOOLS[@]}"; do
            case $tool in
                eksctl)
                    echo "  eksctl: https://eksctl.io/introduction/#installation"
                    ;;
                helm)
                    echo "  helm: https://helm.sh/docs/intro/install/"
                    ;;
                jq)
                    echo "  jq: sudo apt update && sudo apt install -y jq # 或 sudo yum install jq  或 brew install jq"
                    ;;
            esac
        done
        exit 1
    fi
    echo -e "${GREEN}✓${NC} 所有必要工具已安装"
    echo ""
    
    # 检测集群类型
    echo "步骤 3.2: 检测集群类型..."
    CLUSTER_TYPE=$(detect_cluster_type)
    echo ""
    
    # 根据集群类型决定安装方式
    if [ "$CLUSTER_TYPE" == "HYPERPOD" ]; then
        echo -e "${YELLOW}▶${NC} HyperPod 集群不支持使用 eksctl"
        echo "将使用手动方式创建 IAM Role 和 ServiceAccount..."
        echo ""
        USE_EKSCTL=false
    else
        USE_EKSCTL=true
    fi
    
    # 下载 IAM Policy
    echo "步骤 3.3: 下载 IAM Policy..."
    POLICY_VERSION="v2.13.2"
    POLICY_URL="https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/${POLICY_VERSION}/docs/install/iam_policy.json"
    
    if curl -s -f -o iam_policy_alb_full.json $POLICY_URL; then
        echo -e "${GREEN}✓${NC} 已下载 IAM Policy"
    else
        echo -e "${RED}✗${NC} 下载失败"
        exit 1
    fi
    echo ""
    
    # 创建 IAM Policy
    echo "步骤 3.4: 创建 IAM Policy..."
    POLICY_NAME="AWSLoadBalancerControllerIAMPolicy"
    POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
    
    if aws iam get-policy --policy-arn $POLICY_ARN >/dev/null 2>&1; then
        echo -e "${YELLOW}!${NC} Policy 已存在，跳过创建"
    else
        aws iam create-policy \
            --policy-name $POLICY_NAME \
            --policy-document file://iam_policy_alb_full.json \
            --description "IAM policy for AWS Load Balancer Controller" >/dev/null
        echo -e "${GREEN}✓${NC} Policy 已创建"
    fi
    echo ""
    
    # 创建 ServiceAccount 和 IAM Role
    echo "步骤 3.5: 创建 IAM Role 和 ServiceAccount..."
    
    if [ "$USE_EKSCTL" = true ]; then
        # 标准 EKS
        LB_ROLE_NAME="AmazonEKSLoadBalancerControllerRole"
        ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LB_ROLE_NAME}"

        if aws iam get-role --role-name "$LB_ROLE_NAME" >/dev/null 2>&1; then
            echo -e "${YELLOW}!${NC} IAM Role '${LB_ROLE_NAME}' 已存在（可能由其他集群创建）"
            echo "    将复用已有 Role 并更新信任策略..."

            OIDC_URL=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" \
                --query 'cluster.identity.oidc.issuer' --output text)
            OIDC_ID=$(echo "$OIDC_URL" | awk -F'/' '{print $NF}')

            if [ -z "$OIDC_ID" ]; then
                echo -e "${RED}✗${NC} 无法获取当前集群的 OIDC Provider ID"
                exit 1
            fi
            echo -e "${GREEN}✓${NC} 当前集群 OIDC ID: $OIDC_ID"

            cat > trust-policy-lb-update.json << EOFTP
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}:sub": "system:serviceaccount:kube-system:aws-load-balancer-controller",
          "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOFTP
            aws iam update-assume-role-policy \
                --role-name "$LB_ROLE_NAME" \
                --policy-document file://trust-policy-lb-update.json
            echo -e "${GREEN}✓${NC} 已更新 Role 信任策略为当前集群的 OIDC Provider"
            rm -f trust-policy-lb-update.json

            aws iam attach-role-policy \
                --role-name "$LB_ROLE_NAME" \
                --policy-arn "$POLICY_ARN" 2>/dev/null || true

            cat > sa-lb-controller.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aws-load-balancer-controller
  namespace: kube-system
  annotations:
    eks.amazonaws.com/role-arn: ${ROLE_ARN}
EOF
            kubectl apply -f sa-lb-controller.yaml
            rm -f sa-lb-controller.yaml
            echo -e "${GREEN}✓${NC} 已复用已有 Role 并创建 ServiceAccount"
        else
            echo "使用 eksctl 创建 ServiceAccount..."

            if kubectl get sa aws-load-balancer-controller -n kube-system >/dev/null 2>&1; then
                echo -e "${YELLOW}!${NC} ServiceAccount 已存在，将重新创建..."
                eksctl delete iamserviceaccount \
                    --cluster="$CLUSTER_NAME" \
                    --region="$AWS_REGION" \
                    --namespace=kube-system \
                    --name=aws-load-balancer-controller \
                    --wait || true
            fi

            eksctl create iamserviceaccount \
                --cluster="$CLUSTER_NAME" \
                --region="$AWS_REGION" \
                --namespace=kube-system \
                --name=aws-load-balancer-controller \
                --role-name "$LB_ROLE_NAME" \
                --attach-policy-arn="$POLICY_ARN" \
                --approve \
                --override-existing-serviceaccounts
        fi
    else
        # HyperPod - 手动创建
        echo "手动创建 IAM Role 和 ServiceAccount..."
        
        # 获取 OIDC Provider
        OIDC_URL=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' | sed 's|https://||')
        OIDC_PROVIDER=$(aws iam list-open-id-connect-providers --query "OpenIDConnectProviderList[?contains(Arn, '$OIDC_URL')].Arn" --output text | awk -F'/' '{print $NF}' || echo "")
        
        if [ -z "$OIDC_PROVIDER" ]; then
            echo -e "${RED}✗${NC} 无法找到 OIDC Provider"
            echo "请确认集群的 OIDC Provider 已创建"
            exit 1
        fi
        
        echo "OIDC Provider: $OIDC_PROVIDER"
        
        # 创建信任策略
        cat > trust-policy-lb.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_PROVIDER}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_PROVIDER}:sub": "system:serviceaccount:kube-system:aws-load-balancer-controller",
          "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_PROVIDER}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF
        
        # 创建 IAM Role
        ROLE_NAME="AmazonEKSLoadBalancerControllerRole"
        if aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
            echo -e "${YELLOW}!${NC} Role 已存在: $ROLE_NAME"
        else
            aws iam create-role \
                --role-name $ROLE_NAME \
                --assume-role-policy-document file://trust-policy-lb.json \
                --description "IAM Role for AWS Load Balancer Controller"
            echo -e "${GREEN}✓${NC} Role 已创建: $ROLE_NAME"
        fi
        
        # 附加策略
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn "$POLICY_ARN" || true
        
        ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
        
        # 创建 ServiceAccount
        cat > sa-lb-controller.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aws-load-balancer-controller
  namespace: kube-system
  annotations:
    eks.amazonaws.com/role-arn: ${ROLE_ARN}
EOF
        
        kubectl apply -f sa-lb-controller.yaml
        
        # 清理临时文件
        rm -f trust-policy-lb.json sa-lb-controller.yaml
    fi
    
    echo -e "${GREEN}✓${NC} ServiceAccount 和 IAM Role 已配置"
    echo "Role ARN: $ROLE_ARN"
    echo ""
    
    # 修复已有的 IngressClass 所有权（EKS 可能预创建了 IngressClass "alb"，需要让 Helm 接管）
    echo "步骤 3.6: 检查并修复 IngressClass 所有权..."
    if kubectl get ingressclass alb >/dev/null 2>&1; then
        CURRENT_MANAGED_BY=$(kubectl get ingressclass alb -o jsonpath='{.metadata.labels.app\.kubernetes\.io/managed-by}' 2>/dev/null || echo "")
        if [ "$CURRENT_MANAGED_BY" != "Helm" ]; then
            echo -e "${YELLOW}!${NC} 发现已有的 IngressClass \"alb\" (managed-by: ${CURRENT_MANAGED_BY:-未设置})"
            echo "    正在将所有权转移给 Helm..."
            kubectl label ingressclass alb app.kubernetes.io/managed-by=Helm --overwrite
            kubectl annotate ingressclass alb meta.helm.sh/release-name=aws-load-balancer-controller --overwrite
            kubectl annotate ingressclass alb meta.helm.sh/release-namespace=kube-system --overwrite
            echo -e "${GREEN}✓${NC} IngressClass 所有权已转移给 Helm"
        else
            echo -e "${GREEN}✓${NC} IngressClass \"alb\" 已由 Helm 管理，无需修复"
        fi
    else
        echo -e "${BLUE}ℹ${NC}  IngressClass \"alb\" 不存在，将由 Helm 自动创建"
    fi
    echo ""

    # 安装 Controller
    echo "步骤 3.7: 安装 AWS Load Balancer Controller..."
    # helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
    # helm repo update >/dev/null 2>&1
    
    helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller \
        --set region="$AWS_REGION" \
        --set vpcId="$VPC_ID"
    
    echo -e "${GREEN}✓${NC} Controller 已安装"
    echo ""
    
    # 等待 Controller 启动
    echo "步骤 3.8: 等待 Controller 启动..."
    kubectl wait --for=condition=available --timeout=180s \
        deployment/aws-load-balancer-controller -n kube-system
    
    echo -e "${GREEN}✓${NC} Controller 已就绪"
    echo ""
    
    kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
    echo ""
    
    # 清理临时文件
    rm -f iam_policy_alb_full.json
    
    echo ""
    echo -e "${GREEN}✓${NC} AWS Load Balancer Controller 安装完成！"
    echo ""
    
    # 跳过后续的 IAM 修复步骤
    SKIP_IAM_FIX=true
else
    SKIP_IAM_FIX=false
fi

# 如果 Role 不存在且不是安装模式，创建它
if [ "$ROLE_EXISTS" = false ] && [ "$SKIP_IAM_FIX" = false ]; then
    echo "步骤 2.2: 获取 EKS 集群的 OIDC Provider..."
    
    # 获取 OIDC Provider URL
    OIDC_URL=$(aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION --query 'cluster.identity.oidc.issuer' --output text)
    OIDC_ID=$(echo $OIDC_URL | awk -F'/' '{print $NF}')
    OIDC_PROVIDER="oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}"
    
    echo -e "${GREEN}✓${NC} OIDC Provider: $OIDC_PROVIDER"
    echo ""
    
    echo "步骤 2.3: 创建 IAM Role 信任策略..."
    cat > trust-policy.json << EOF
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
          "${OIDC_PROVIDER}:sub": "system:serviceaccount:kube-system:aws-load-balancer-controller",
          "${OIDC_PROVIDER}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF
    
    echo "步骤 2.4: 创建 IAM Role..."
    if aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json \
        --description "IAM Role for AWS Load Balancer Controller in SageMaker HyperPod" \
        >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} IAM Role 创建成功: $ROLE_NAME"
    else
        echo -e "${RED}✗${NC} 创建 IAM Role 失败"
        echo "请检查您是否有足够的 IAM 权限"
        rm -f trust-policy.json
        exit 1
    fi
    
    # 清理临时文件
    rm -f trust-policy.json
    echo ""
fi

# 如果已经安装了 Controller，跳过 IAM 修复
if [ "$SKIP_IAM_FIX" = true ]; then
    # 跳转到最终验证
    echo "跳过 IAM 修复步骤..."
else
    # 下载官方 IAM Policy
    echo "步骤 2.5: 下载官方 IAM Policy..."
    POLICY_VERSION="v2.13.2"  # 与当前运行的 Controller 版本一致
    POLICY_URL="https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/${POLICY_VERSION}/docs/install/iam_policy.json"

    if curl -s -f -o iam_policy_alb_full.json $POLICY_URL; then
        echo -e "${GREEN}✓${NC} 成功下载 IAM Policy (版本 $POLICY_VERSION)"
    else
        echo -e "${RED}✗${NC} 下载 IAM Policy 失败"
        exit 1
    fi
    echo ""

# 检查现有策略
echo "步骤 2.6: 检查现有的 IAM 策略..."
POLICY_NAME="AWSLoadBalancerControllerIAMPolicy"
EXISTING_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"

echo "检查托管策略是否已存在..."
if aws iam get-policy --policy-arn $EXISTING_POLICY_ARN >/dev/null 2>&1; then
    echo -e "${YELLOW}!${NC} 策略已存在: $POLICY_NAME"
    echo "    将创建新版本..."
    
    # 获取当前所有版本
    VERSIONS_COUNT=$(aws iam list-policy-versions --policy-arn $EXISTING_POLICY_ARN --query 'length(Versions)' --output text)
    
    # AWS 限制最多 5 个版本，如果已经有 5 个，删除最老的非默认版本
    if [ "$VERSIONS_COUNT" -ge 5 ]; then
        echo "    策略版本已达上限，删除最旧的非默认版本..."
        OLDEST_VERSION=$(aws iam list-policy-versions --policy-arn $EXISTING_POLICY_ARN \
            --query 'Versions[?IsDefaultVersion==`false`]|[0].VersionId' --output text)
        if [ "$OLDEST_VERSION" != "None" ] && [ -n "$OLDEST_VERSION" ]; then
            aws iam delete-policy-version --policy-arn $EXISTING_POLICY_ARN --version-id $OLDEST_VERSION
            echo -e "${GREEN}✓${NC} 已删除旧版本: $OLDEST_VERSION"
        fi
    fi
    
    # 创建新版本并设置为默认
    NEW_VERSION=$(aws iam create-policy-version \
        --policy-arn $EXISTING_POLICY_ARN \
        --policy-document file://iam_policy_alb_full.json \
        --set-as-default \
        --query 'PolicyVersion.VersionId' \
        --output text)
    echo -e "${GREEN}✓${NC} 已创建新版本: $NEW_VERSION (已设为默认)"
    POLICY_ARN=$EXISTING_POLICY_ARN
else
    echo "策略不存在，将创建新策略..."
    POLICY_ARN=$(aws iam create-policy \
        --policy-name $POLICY_NAME \
        --policy-document file://iam_policy_alb_full.json \
        --description "IAM policy for AWS Load Balancer Controller" \
        --query 'Policy.Arn' \
        --output text)
    echo -e "${GREEN}✓${NC} 已创建新策略: $POLICY_ARN"
fi
echo ""

# 检查策略是否已附加到 Role
echo "步骤 2.7: 检查策略是否已附加到 Role..."
if aws iam list-attached-role-policies --role-name $ROLE_NAME --query "AttachedPolicies[?PolicyArn=='$POLICY_ARN']" --output text | grep -q "$POLICY_NAME"; then
    echo -e "${YELLOW}!${NC} 策略已附加到 Role"
    echo "    策略内容已更新为最新版本"
else
    echo "附加策略到 Role..."
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn $POLICY_ARN
    echo -e "${GREEN}✓${NC} 策略已成功附加到 Role"
fi
echo ""

# 显示当前附加的所有策略
echo "步骤 2.8: 验证 Role 的所有附加策略..."
echo "当前附加到 $ROLE_NAME 的策略:"
aws iam list-attached-role-policies --role-name $ROLE_NAME --query 'AttachedPolicies[].[PolicyName,PolicyArn]' --output table
echo ""

# 等待权限生效
echo "步骤 2.9: 等待 IAM 权限生效..."
echo "AWS IAM 权限通常在几秒钟内生效，但可能需要最多 60 秒..."
for i in {10..1}; do
    echo -ne "\r等待中... $i 秒  "
    sleep 1
done
echo -e "\r${GREEN}✓${NC} 等待完成          "
echo ""

# 重启 Controller 以确保使用新权限
echo "步骤 2.10: 重启 Load Balancer Controller..."
echo "重启 Controller 以确保 Pod 获取新的 IAM credentials..."
if kubectl rollout restart deployment hyperpod-inference-operator-alb -n kube-system >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Controller 已触发重启"
    
    # 等待新 Pods 启动
    echo ""
    echo "等待新 Pods 启动..."
    for i in {15..1}; do
        echo -ne "\r等待中... $i 秒  "
        sleep 1
    done
    echo -e "\r${GREEN}✓${NC} 等待完成          "
    
    # 等待 Pods 就绪
    echo ""
    echo "等待 Pods 就绪..."
    if kubectl rollout status deployment hyperpod-inference-operator-alb -n kube-system --timeout=60s >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Controller Pods 已就绪"
    else
        echo -e "${YELLOW}!${NC} 等待超时，但 Pods 可能仍在启动中"
    fi
else
    echo -e "${YELLOW}!${NC} 无法重启 Controller，请手动执行:"
    echo "   kubectl rollout restart deployment hyperpod-inference-operator-alb -n kube-system"
fi
echo ""

fi  # 结束 SKIP_IAM_FIX 判断

# 最终验证
echo ""
echo "=========================================="
echo "阶段 4: 验证和测试"
echo "=========================================="
echo ""

# 检查 Controller 日志
echo "步骤 4.1: 检查 Load Balancer Controller 状态..."
echo "Controller Pods 状态:"
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller -o wide
echo ""

echo "最近的 Controller 日志 (检查是否还有权限错误):"
echo "----------------------------------------"
kubectl logs -n kube-system deployment/aws-load-balancer-controller --tail=10 --prefix=true
echo "----------------------------------------"
echo ""

# 检查 Service 状态
echo "步骤 4.2: 检查 Service 状态..."
if kubectl get svc whisper-triton-unified-nlb -n default >/dev/null 2>&1; then
    echo "Service: whisper-triton-unified-nlb"
    kubectl get svc whisper-triton-unified-nlb -n default
    echo ""
    
    EXTERNAL_IP=$(kubectl get svc whisper-triton-unified-nlb -n default -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" == "null" ]; then
        echo -e "${YELLOW}!${NC} EXTERNAL-IP 仍为 pending"
        echo ""
        echo "建议操作:"
        echo "1. 等待 1-2 分钟后查看日志:"
        echo "   kubectl logs -n kube-system deployment/hyperpod-inference-operator-alb --tail=50 -f"
        echo ""
        echo "2. 查看 Service 事件:"
        echo "   kubectl describe svc whisper-triton-unified-nlb -n default"
        echo ""
        echo "3. 如果问题持续，检查 Controller 是否看到了新的权限:"
        echo "   kubectl rollout restart deployment hyperpod-inference-operator-alb -n kube-system"
    else
        echo -e "${GREEN}✓${NC} Load Balancer 已创建成功!"
        echo "   DNS: $EXTERNAL_IP"
    fi
else
    echo -e "${YELLOW}!${NC} Service 'whisper-triton-unified-nlb' 未找到"
    echo "   您可以创建 Service 来测试 Load Balancer Controller"
fi
echo ""

# 清理临时文件
echo "步骤 4.3: 清理临时文件..."
rm -f iam_policy_alb_full.json trust-policy.json
echo -e "${GREEN}✓${NC} 已删除临时文件"
echo ""

==========================================
阶段 5: 等待和验证 LoadBalancer 创建
==========================================

步骤 5.1: 检查现有 LoadBalancer 服务...
if kubectl get svc whisper-triton-unified-nlb >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} 发现 LoadBalancer 服务: whisper-triton-unified-nlb"
    
    echo "步骤 5.2: 等待 External IP 分配..."
    echo -e "${YELLOW}▶${NC} 等待 LoadBalancer 分配 External IP (最多等待 5 分钟)..."
    
    timeout=300  # 5 分钟
    counter=0
    
    while [ $counter -lt $timeout ]; do
        external_ip=$(kubectl get svc whisper-triton-unified-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
        
        if [ -n "$external_ip" ] && [ "$external_ip" != "<none>" ] && [ "$external_ip" != "null" ]; then
            echo -e "${GREEN}✓${NC} External IP 已分配: $external_ip"
            break
        fi
        
        # 每30秒显示一次进度
        if [ $((counter % 30)) -eq 0 ]; then
            echo -e "${BLUE}ℹ${NC} 等待中... (已等待 ${counter}s)"
            # 显示当前状态
            kubectl get svc whisper-triton-unified-nlb --no-headers | awk '{print "   当前状态: " $4}'
        fi
        
        sleep 5
        counter=$((counter + 5))
    done
    
    if [ $counter -ge $timeout ]; then
        echo -e "${RED}✗${NC} 超时：LoadBalancer 在 5 分钟内未获得 External IP"
        echo ""
        echo "可能的原因和解决方案:"
        echo "1. 检查 AWS Load Balancer Controller 日志:"
        echo "   kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller --tail=20"
        echo ""
        echo "2. 检查 Service 事件:"
        echo "   kubectl describe svc whisper-triton-unified-nlb"
        echo ""
        echo "3. 重启 Controller (如果有 webhook 错误):"
        echo "   kubectl rollout restart deployment/aws-load-balancer-controller -n kube-system"
        echo ""
    else
        echo "步骤 5.3: 验证 LoadBalancer 功能..."
        echo -e "${YELLOW}▶${NC} 测试 LoadBalancer 连通性..."
        
        # 等待几秒让 LoadBalancer 完全就绪
        sleep 10
        
        # 测试健康检查端点
        if curl -s --connect-timeout 10 "http://$external_ip:8080/ping" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} API 服务 (端口 8080) 响应正常"
        else
            echo -e "${YELLOW}!${NC} API 服务 (端口 8080) 暂时无响应 (可能仍在启动中)"
        fi
        
        echo ""
        echo -e "${GREEN}✓${NC} LoadBalancer 部署成功！"
        echo ""
        echo "服务端点信息:"
        echo "  External URL: http://$external_ip:8080"
        echo "  Triton URL:   http://$external_ip:10086"
        echo ""
        echo "测试命令:"
        echo "  # API 健康检查"
        echo "  curl http://$external_ip:8080/ping"
        echo ""
        echo "  # Triton 健康检查"  
        echo "  curl http://$external_ip:10086/v2/health/ready"
        echo ""
        echo "  # 音频转录测试 (需要 audio.wav 文件)"
        echo "  python3 -c \""
        echo "import requests, base64"
        echo "with open('audio.wav', 'rb') as f:"
        echo "    audio_b64 = base64.b64encode(f.read()).decode('utf-8')"
        echo "url = 'http://$external_ip:8080/invocations'"
        echo "payload = {'audio_data': audio_b64, 'whisper_prompt': ''}"
        echo "response = requests.post(url, json=payload)"
        echo "print(f'Status: {response.status_code}')"
        echo "print(f'Result: {response.json()}')\""
        echo ""
    fi
else
    echo -e "${YELLOW}!${NC} 未发现 whisper-triton-unified-nlb 服务"
    echo "请先部署 Whisper Triton 服务，然后再运行此脚本"
fi

echo ""
echo "=========================================="
echo "AWS Load Balancer Controller 配置完成！"
echo "=========================================="

# 创建增强的状态检查脚本
cat > check_lb_status.sh << 'EOF'
#!/bin/bash
# 增强的 Load Balancer 状态检查脚本

echo "=== AWS Load Balancer Controller 状态 ==="
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller

echo ""
echo "=== LoadBalancer Service 状态 ==="
kubectl get svc whisper-triton-unified-nlb -n default

echo ""
echo "=== Service 详细信息 ==="
kubectl describe svc whisper-triton-unified-nlb -n default | grep -A 10 "Events:"

echo ""
echo "=== Controller 最近日志 ==="
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller --tail=10 | grep -E "error|Error|whisper|registered|reconcile"

echo ""
echo "=== 快速连通性测试 ==="
external_ip=$(kubectl get svc whisper-triton-unified-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
if [ -n "$external_ip" ] && [ "$external_ip" != "<none>" ] && [ "$external_ip" != "null" ]; then
    echo "External IP: $external_ip"
    echo "测试 API 端点..."
    if curl -s --connect-timeout 5 "http://$external_ip:8080/ping" >/dev/null 2>&1; then
        echo "✓ API 服务 (8080) 可访问"
    else
        echo "✗ API 服务 (8080) 无响应"
    fi
else
    echo "External IP 尚未分配"
fi
EOF

chmod +x check_lb_status.sh
echo -e "${GREEN}✓${NC} 已创建增强的状态检查脚本: check_lb_status.sh"
echo "   运行 ./check_lb_status.sh 可快速查看完整状态"

