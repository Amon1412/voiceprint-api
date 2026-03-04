# 3D-Speaker 声纹识别API

基于3D-Speaker模型的声纹识别服务，提供声纹注册、识别、删除等功能。

目前用于xiaozhi说话人识别，[xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server)

## 🛠️ 安装和配置

### 1. 安装依赖
```bash
conda remove -n voiceprint-api --all -y
conda create -n voiceprint-api python=3.10 -y
conda activate voiceprint-api
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip install -r requirements.txt
```

### 2. 数据库配置
创建MySQL数据库和表：
```sql
CREATE DATABASE voiceprint_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE voiceprint_db;

CREATE TABLE voiceprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    speaker_id VARCHAR(255) NOT NULL UNIQUE,
    feature_vector LONGBLOB NOT NULL,
    cluster_count INT DEFAULT 1 COMMENT '聚类音频数量',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_speaker_id (speaker_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 4. 配置文件
复制voiceprint.yaml到data目录，并编辑 `data/.voiceprint.yaml`：
```yaml
mysql:
  host: "127.0.0.1"
  port: 3306
  user: "root"
  password: "your_password"
  database: "voiceprint_db"
```

## 🚀 启动服务

### 开发环境
```bash
python -m app.main
```

### 生产环境
```bash
python start_server.py
```

## 📚 API文档

启动服务后，访问以下地址查看API文档：
- Swagger UI: http://localhost:8005/voiceprint/docs
