import argparse
import boto3
import os

def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key

def download_folder_from_s3(source_s3_url, local_dir_path):
    s3 = boto3.client('s3')
    print(f"s3path: {source_s3_url}")
    print(f"download_folder: {local_dir_path}")
    bucket_name, s3_folder_path = get_bucket_and_key(source_s3_url)

    # 确保本地目录存在
    os.makedirs(local_dir_path, exist_ok=True)

    # 列出 S3 文件夹中的所有对象
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path):
        for obj in page.get('Contents', []):
            # 获取文件的相对路径
            relative_path = obj['Key'][len(s3_folder_path):]
            # 构建本地文件路径
            local_file_path = os.path.join(local_dir_path, relative_path)
            # 确保本地文件夹存在
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # 下载文件
            try:
                s3.download_file(bucket_name, obj['Key'], local_file_path)
                print(f'文件 {obj["Key"]} 已下载到 {local_file_path}')
            except Exception as e:
                print(f'下载 {obj["Key"]} 失败: {e}')

def main():
    parser = argparse.ArgumentParser(description="Download a folder from S3 to a local directory.")
    parser.add_argument("--source_s3_url", type=str, required=True, help="Source S3 URL (e.g., s3://bucket-name/folder/)")
    parser.add_argument("--local_dir_path", type=str, default="model_repo_whisper_trtllm", required=True, help="Local directory path to save the downloaded files")
    parser.add_argument("--working_dir", type=str, default="/workspace", help="Working directory (default: /workspace)")

    args = parser.parse_args()

    account_id = boto3.client('sts').get_caller_identity().get('Account')
    my_session = boto3.session.Session()
    region = my_session.region_name

    print(f"Account ID: {account_id}")
    print(f"Region: {region}")

    local_dir_path = os.path.join(args.working_dir, args.local_dir_path)

    download_folder_from_s3(args.source_s3_url, local_dir_path)

if __name__ == "__main__":
    main()
