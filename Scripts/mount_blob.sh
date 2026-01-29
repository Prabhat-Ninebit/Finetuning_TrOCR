# sudo apt update
# sudo apt install blobfuse2 -y
# mkdir -p ~/.blobfuse
# nano ~/.blobfuse/config.yaml
# sudo mkdir -p /mnt/blob
# sudo chown azureuser:azureuser /mnt/blob

# # Create mount point
# sudo mkdir -p /mnt/blob
# sudo chown azureuser:azureuser /mnt/blob


# # mounting Blob Storage
# blobfuse2 mount /mnt/blob --config-file ~/.blobfuse/config.yaml
# echo "Blob storage mounted at /mnt/blob"


#!/bin/bash
set -e

echo "🔹 Updating system & installing blobfuse2..."
sudo apt update
sudo apt install -y blobfuse2

echo "🔹 Creating blobfuse config directory..."
mkdir -p ~/.blobfuse

# NOTE: config.yaml should already exist
# nano ~/.blobfuse/config.yaml

echo "🔹 Creating mount, cache, and temp directories..."
sudo mkdir -p /mnt/blob
sudo mkdir -p /mnt/blobfuse_cache
sudo mkdir -p /mnt/blobfuse_tmp

echo "🔹 Fixing ownership..."
sudo chown -R azureuser:azureuser /mnt/blob
sudo chown -R azureuser:azureuser /mnt/blobfuse_cache
sudo chown -R azureuser:azureuser /mnt/blobfuse_tmp

echo "🔹 Fixing permissions..."
sudo chmod 700 /mnt/blobfuse_cache /mnt/blobfuse_tmp

echo "🔹 Cleaning cache & temp directories..."
rm -rf /mnt/blobfuse_cache/*
rm -rf /mnt/blobfuse_tmp/*

echo "🔹 Unmounting existing mount (if any)..."
fusermount3 -u /mnt/blob 2>/dev/null || true

echo "🔹 Mounting Azure Blob Storage..."
blobfuse2 mount /mnt/blob --config-file ~/.blobfuse/config.yaml

echo "✅ Blob storage mounted at /mnt/blob"
