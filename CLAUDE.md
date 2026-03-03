# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 3D-Speaker 模型的声纹识别 API 服务，用于 xiaozhi-esp32-server 的说话人识别。技术栈：FastAPI + ModelScope (PyTorch) + MySQL。

## Commands

### 开发环境启动
```bash
python -m app.main
```
启用热重载，监听 0.0.0.0:8005。

### 生产环境启动
```bash
python start_server.py
```

### Docker 构建与运行
```bash
docker compose up -d
```
数据卷挂载 `./data:/app/data` 用于持久化配置文件。

### 安装依赖
```bash
conda create -n voiceprint-api python=3.10 -y
conda activate voiceprint-api
pip install -r requirements.txt
```

### 数据库初始化
需要先创建 MySQL 数据库 `voiceprint_db`，建表 SQL 见 README.md。

## Architecture

### 请求处理流程
```
HTTP Request → FastAPI (app/application.py)
  → Bearer Token 认证 (app/api/dependencies.py → app/core/security.py)
  → 路由 (app/api/v1/api.py)
  → 端点处理 (app/api/v1/voiceprint.py)
  → 业务逻辑 (app/services/voiceprint_service.py)
  → 数据库操作 (app/database/voiceprint_db.py)
```

### 核心模块职责
- **app/application.py** — FastAPI 应用创建，CORS/文档路由配置，应用入口为 `app` 对象
- **app/services/voiceprint_service.py** — 核心业务逻辑，加载 3D-Speaker 模型 (`iic/speech_campplus_sv_zh-cn_3dspeaker_16k`)，声纹提取与相似度计算
- **app/database/connection.py** — MySQL 连接池管理，PyMySQL context manager
- **app/database/voiceprint_db.py** — voiceprints 表 CRUD，特征向量以 numpy 二进制 (LONGBLOB) 存储
- **app/core/config.py** — YAML 配置加载，配置文件路径为 `data/.voiceprint.yaml`，首次运行自动生成 UUID token
- **app/utils/audio_utils.py** — 音频重采样到 16kHz、格式验证（0.5-30秒，≥8kHz）

### 关键设计约束
- **单 worker 模式** — Uvicorn 必须 `workers=1`，避免 PyTorch 模型重复加载占用显存
- **模型推理线程锁** — `VoiceprintService` 使用 `threading.Lock` 保证推理线程安全
- **模型预热** — 启动时 `_warmup_model()` 预热多种采样率的测试音频，消除首次推理延迟
- **GPU 自动检测** — 优先使用 CUDA，不可用时回退 CPU

### 配置体系
配置文件模板为项目根目录 `voiceprint.yaml`，运行时读取 `data/.voiceprint.yaml`。主要配置项：
- `server.host/port/authorization` — 服务地址与 API token
- `mysql.*` — 数据库连接信息
- 相似度阈值默认 0.2，目标采样率 16000Hz

### API 端点
所有端点挂载在 `/voiceprint` 前缀下，Bearer Token 认证：
- `POST /voiceprint/register` — 注册声纹（speaker_id + WAV 文件）
- `POST /voiceprint/identify` — 识别说话人（speaker_ids 逗号分隔 + WAV 文件）
- `DELETE /voiceprint/{speaker_id}` — 删除声纹
- `GET /voiceprint/health?key={token}` — 健康检查（query param 认证）

聚类端点（`/voiceprint/cluster` 前缀）：
- `POST /voiceprint/cluster` — 创建聚类任务（传本地文件路径数组，异步处理）
- `GET /voiceprint/cluster/{task_id}` — 查询聚类进度/结果
- `POST /voiceprint/cluster/{task_id}/confirm` — 确认聚类并注册质心声纹
- `DELETE /voiceprint/cluster/{task_id}` — 删除聚类任务

### 聚类模块
- **app/services/cluster_service.py** — 聚类任务管理器（`cluster_manager` 单例），后台线程异步处理，scipy 层次聚类，纯内存状态
- **app/api/v1/cluster.py** — 聚类 API 端点
- **app/models/cluster.py** — 聚类请求/响应 Pydantic 模型
- 聚类流程：小智传文件路径数组 → 逐文件提取向量 → 层次聚类 → 返回分组 → 用户确认角色 → 取质心注册

### 日志系统
使用 loguru 替代标准 logging，配置在 `app/core/logger.py`。日志文件输出到 `logs/` 目录，10MB 轮转，7天保留。自定义方法：`logger.success()`, `logger.fail()`, `logger.start()`, `logger.complete()`。

## Conventions

- 服务层使用模块级单例实例：`voiceprint_service`, `voiceprint_db`, `audio_processor`, `cluster_manager`
- 仅接受 WAV 格式音频上传
- 数据库操作使用 context manager 模式获取游标
- 所有代码注释和日志使用中文
