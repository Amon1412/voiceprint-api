from pydantic import BaseModel, Field
from typing import List, Optional


class ClusterCreateRequest(BaseModel):
    """聚类任务创建请求"""

    file_paths: List[str] = Field(..., description="音频文件路径列表")
    similarity_threshold: Optional[float] = Field(
        None, description="聚类相似度阈值（0-1），默认读配置"
    )

    class Config:
        schema_extra = {
            "example": {
                "file_paths": ["/data/audio/001.wav", "/data/audio/002.wav"],
                "similarity_threshold": 0.7,
            }
        }


class InvalidFileInfo(BaseModel):
    """无效文件信息"""

    file_path: str
    reason: str


class ClusterCreateResponse(BaseModel):
    """聚类任务创建响应"""

    task_id: str
    total_files: int
    valid_files: int
    invalid_files: List[InvalidFileInfo]


class ClusterFileInfo(BaseModel):
    """聚类中的文件信息"""

    file_path: str
    distance_to_centroid: float


class ClusterInfo(BaseModel):
    """单个聚类信息"""

    cluster_id: int
    file_count: int
    centroid_file: str = Field(description="距离质心最近的文件路径")
    files: List[ClusterFileInfo]


class OutlierFileInfo(BaseModel):
    """离群点文件信息"""

    file_path: str


class ClusterStats(BaseModel):
    """聚类统计信息"""

    total_files: int
    valid_files: int
    total_clusters: int
    outlier_count: int
    processing_time_seconds: float


class ClusterStatusResponse(BaseModel):
    """聚类任务状态/结果响应"""

    task_id: str
    status: str = Field(description="任务状态: processing, completed, failed")
    total_files: int
    processed_files: int
    progress_percent: float
    error: Optional[str] = None
    # 以下字段仅在 status=completed 时返回
    clusters: Optional[List[ClusterInfo]] = None
    outliers: Optional[List[OutlierFileInfo]] = None
    stats: Optional[ClusterStats] = None


class ClusterAssignment(BaseModel):
    """聚类角色分配"""

    cluster_id: int
    speaker_id: str


class ClusterConfirmRequest(BaseModel):
    """聚类确认请求"""

    assignments: List[ClusterAssignment]

    class Config:
        schema_extra = {
            "example": {
                "assignments": [
                    {"cluster_id": 0, "speaker_id": "张三"},
                    {"cluster_id": 2, "speaker_id": "李四"},
                ]
            }
        }


class ClusterRegisteredInfo(BaseModel):
    """注册成功的信息"""

    speaker_id: str
    cluster_id: int
    file_count: int


class ClusterFailedInfo(BaseModel):
    """注册失败的信息"""

    speaker_id: str
    cluster_id: int
    reason: str


class ClusterConfirmResponse(BaseModel):
    """聚类确认响应"""

    success: bool
    registered: List[ClusterRegisteredInfo]
    failed: List[ClusterFailedInfo]
