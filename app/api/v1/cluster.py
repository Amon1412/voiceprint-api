from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer
import time
import json
from typing import List, Optional
from ...models.cluster import (
    ClusterCreateRequest,
    ClusterCreateResponse,
    ClusterStatusResponse,
    ClusterConfirmRequest,
    ClusterConfirmResponse,
    InvalidFileInfo,
    ClusterInfo,
    ClusterFileInfo,
    OutlierFileInfo,
    ClusterStats,
    ClusterRegisteredInfo,
    ClusterFailedInfo,
    ClusterUploadResponse,
    ClusterUploadInfo,
    ClusterUploadFileInfo,
    ClusterUploadStatusResponse,
    ClusterMergeConfirmRequest,
    ClusterMergeConfirmResponse,
    ClusterMergeRegisteredInfo,
)
from ...services.cluster_service import cluster_manager
from ...api.dependencies import AuthorizationToken
from ...core.config import settings
from ...core.logger import get_logger

# 创建安全模式
security = HTTPBearer(description="接口令牌")

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "",
    summary="创建聚类任务",
    response_model=ClusterCreateResponse,
    description="传入音频文件路径列表，创建聚类任务",
    dependencies=[Depends(security)],
)
async def create_cluster_task(
    token: AuthorizationToken,
    request: ClusterCreateRequest,
):
    """
    创建声纹聚类任务

    Args:
        token: 接口令牌（Header）
        request: 聚类请求（文件路径列表和可选阈值）

    Returns:
        ClusterCreateResponse: 任务创建结果
    """
    start_time = time.time()
    logger.info(
        f"收到聚类任务请求，文件数: {len(request.file_paths)}"
    )

    try:
        # 检查文件数限制
        if len(request.file_paths) > settings.cluster_max_files:
            raise HTTPException(
                status_code=400,
                detail=f"文件数超过限制，最大{settings.cluster_max_files}个",
            )

        if len(request.file_paths) < 2:
            raise HTTPException(
                status_code=400,
                detail="至少需要2个音频文件",
            )

        task_id, total_files, valid_files, invalid_files = (
            cluster_manager.create_task(
                request.file_paths, request.similarity_threshold
            )
        )

        total_time = time.time() - start_time
        logger.info(
            f"聚类任务创建完成: {task_id}, 耗时: {total_time:.3f}秒"
        )

        return ClusterCreateResponse(
            task_id=task_id,
            total_files=total_files,
            valid_files=valid_files,
            invalid_files=[
                InvalidFileInfo(**f) for f in invalid_files
            ],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建聚类任务异常: {e}")
        raise HTTPException(
            status_code=500, detail=f"创建聚类任务失败: {str(e)}"
        )


@router.get(
    "/{task_id}",
    summary="查询聚类任务状态",
    response_model=ClusterStatusResponse,
    description="查询聚类任务的处理进度和结果",
    dependencies=[Depends(security)],
)
async def get_cluster_status(
    token: AuthorizationToken,
    task_id: str,
):
    """
    查询聚类任务状态

    Args:
        token: 接口令牌（Header）
        task_id: 任务ID

    Returns:
        ClusterStatusResponse: 任务状态和结果
    """
    try:
        task = cluster_manager.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=404, detail=f"任务不存在: {task_id}"
            )

        # 构建响应
        clusters = None
        outliers = None
        stats = None

        if task.get("clusters") is not None:
            clusters = [
                ClusterInfo(
                    cluster_id=c["cluster_id"],
                    file_count=c["file_count"],
                    centroid_file=c["centroid_file"],
                    files=[ClusterFileInfo(**f) for f in c["files"]],
                )
                for c in task["clusters"]
            ]
            outliers = [
                OutlierFileInfo(**o) for o in task.get("outliers", [])
            ]
            stats = ClusterStats(**task["stats"])

        return ClusterStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            total_files=task["total_files"],
            processed_files=task["processed_files"],
            progress_percent=task["progress_percent"],
            error=task.get("error"),
            clusters=clusters,
            outliers=outliers,
            stats=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询聚类任务异常: {e}")
        raise HTTPException(
            status_code=500, detail=f"查询聚类任务失败: {str(e)}"
        )


@router.post(
    "/{task_id}/confirm",
    summary="确认聚类结果并注册声纹",
    response_model=ClusterConfirmResponse,
    description="为选定的聚类分配角色名，计算质心并注册为声纹",
    dependencies=[Depends(security)],
)
async def confirm_clusters(
    token: AuthorizationToken,
    task_id: str,
    request: ClusterConfirmRequest,
):
    """
    确认聚类结果并注册声纹

    Args:
        token: 接口令牌（Header）
        task_id: 任务ID
        request: 聚类确认请求（角色分配列表）

    Returns:
        ClusterConfirmResponse: 注册结果
    """
    start_time = time.time()
    logger.info(
        f"收到聚类确认请求: {task_id}, "
        f"分配数: {len(request.assignments)}"
    )

    try:
        result = cluster_manager.confirm_clusters(
            task_id,
            [a.dict() for a in request.assignments],
        )

        total_time = time.time() - start_time
        logger.info(
            f"聚类确认完成: {task_id}, "
            f"注册: {len(result['registered'])}, "
            f"失败: {len(result['failed'])}, "
            f"耗时: {total_time:.3f}秒"
        )

        return ClusterConfirmResponse(
            success=result["success"],
            registered=[
                ClusterRegisteredInfo(**r) for r in result["registered"]
            ],
            failed=[
                ClusterFailedInfo(**f) for f in result["failed"]
            ],
        )

    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"任务不存在: {task_id}"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"聚类确认异常: {e}")
        raise HTTPException(
            status_code=500, detail=f"聚类确认失败: {str(e)}"
        )


@router.post(
    "/upload",
    summary="上传音频创建增量聚类任务",
    response_model=ClusterUploadResponse,
    description="上传音频文件和音频ID，支持已有说话人质心作为锚点参与聚类",
    dependencies=[Depends(security)],
)
async def create_upload_cluster_task(
    token: AuthorizationToken,
    files: List[UploadFile] = File(..., description="音频文件列表"),
    audio_ids: str = Form(..., description="音频ID列表，JSON数组格式"),
    existing_speaker_ids: Optional[str] = Form(None, description="已有说话人ID列表，JSON数组格式"),
    similarity_threshold: Optional[float] = Form(None, description="聚类相似度阈值"),
    session_groups: Optional[str] = Form(None, description="session分组映射，JSON对象格式 {sessionId: [audioId, ...]}"),
):
    """
    上传音频文件创建增量聚类任务

    - files: multipart上传的音频文件
    - audio_ids: 对应的音频ID（与files一一对应）
    - existing_speaker_ids: 已有说话人ID（其质心作为锚点参与聚类）
    - similarity_threshold: 可选的相似度阈值
    """
    start_time = time.time()

    try:
        # 解析audio_ids
        try:
            audio_id_list = json.loads(audio_ids)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="audio_ids格式错误，应为JSON数组")

        if len(files) != len(audio_id_list):
            raise HTTPException(
                status_code=400,
                detail=f"文件数({len(files)})与音频ID数({len(audio_id_list)})不匹配"
            )

        if len(files) > settings.cluster_max_files:
            raise HTTPException(
                status_code=400,
                detail=f"文件数超过限制，最大{settings.cluster_max_files}个"
            )

        if len(files) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个音频文件")

        # 解析existing_speaker_ids
        speaker_ids = None
        if existing_speaker_ids:
            try:
                speaker_ids = json.loads(existing_speaker_ids)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="existing_speaker_ids格式错误")

        # 解析session_groups
        session_group_map = None
        if session_groups:
            try:
                session_group_map = json.loads(session_groups)
            except json.JSONDecodeError:
                logger.warning("session_groups格式错误，将使用普通聚类")

        # 读取音频数据
        audio_items = []
        for i, f in enumerate(files):
            content = await f.read()
            audio_items.append((audio_id_list[i], content))

        logger.info(
            f"收到增量聚类请求，音频数: {len(audio_items)}, "
            f"已有说话人: {len(speaker_ids) if speaker_ids else 0}"
        )

        task_id, total_files, valid_files, invalid_files = (
            cluster_manager.create_task_from_uploads(
                audio_items, speaker_ids, similarity_threshold, session_group_map
            )
        )

        total_time = time.time() - start_time
        logger.info(f"增量聚类任务创建完成: {task_id}, 耗时: {total_time:.3f}秒")

        return ClusterUploadResponse(
            task_id=task_id,
            total_files=total_files,
            valid_files=valid_files,
            invalid_files=[InvalidFileInfo(**f) for f in invalid_files],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建增量聚类任务异常: {e}")
        raise HTTPException(status_code=500, detail=f"创建增量聚类任务失败: {str(e)}")


@router.get(
    "/upload/{task_id}",
    summary="查询增量聚类任务状态",
    response_model=ClusterUploadStatusResponse,
    description="查询增量聚类任务的处理进度和结果",
    dependencies=[Depends(security)],
)
async def get_upload_cluster_status(
    token: AuthorizationToken,
    task_id: str,
):
    """查询增量聚类任务状态"""
    try:
        task = cluster_manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        clusters = None
        outliers = None
        stats = None

        if task.get("clusters") is not None:
            clusters = [
                ClusterUploadInfo(
                    cluster_id=c["cluster_id"],
                    file_count=c["file_count"],
                    files=[ClusterUploadFileInfo(**f) for f in c["files"]],
                    existing_speaker_id=c.get("existing_speaker_id"),
                    existing_speaker_count=c.get("existing_speaker_count"),
                )
                for c in task["clusters"]
            ]
            outliers = task.get("outliers", [])
            stats = ClusterStats(**task["stats"])

        return ClusterUploadStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            total_files=task["total_files"],
            processed_files=task["processed_files"],
            progress_percent=task["progress_percent"],
            error=task.get("error"),
            clusters=clusters,
            outliers=outliers,
            stats=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询增量聚类任务异常: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post(
    "/upload/{task_id}/confirm",
    summary="确认增量聚类结果（支持合并）",
    response_model=ClusterMergeConfirmResponse,
    description="确认增量聚类结果，支持与已有说话人合并质心",
    dependencies=[Depends(security)],
)
async def confirm_upload_clusters(
    token: AuthorizationToken,
    task_id: str,
    request: ClusterMergeConfirmRequest,
):
    """确认增量聚类结果并注册/合并声纹"""
    start_time = time.time()
    logger.info(
        f"收到增量聚类确认请求: {task_id}, 分配数: {len(request.assignments)}"
    )

    try:
        result = cluster_manager.confirm_clusters_merge(
            task_id,
            [a.dict() for a in request.assignments],
        )

        total_time = time.time() - start_time
        logger.info(
            f"增量聚类确认完成: {task_id}, "
            f"注册: {len(result['registered'])}, "
            f"失败: {len(result['failed'])}, "
            f"耗时: {total_time:.3f}秒"
        )

        return ClusterMergeConfirmResponse(
            success=result["success"],
            registered=[
                ClusterMergeRegisteredInfo(**r) for r in result["registered"]
            ],
            failed=[
                ClusterFailedInfo(**f) for f in result["failed"]
            ],
        )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增量聚类确认异常: {e}")
        raise HTTPException(status_code=500, detail=f"确认失败: {str(e)}")


@router.delete(
    "/{task_id}",
    summary="删除聚类任务",
    description="删除指定的聚类任务及其内存数据",
    dependencies=[Depends(security)],
)
async def delete_cluster_task(
    token: AuthorizationToken,
    task_id: str,
):
    """
    删除聚类任务

    Args:
        token: 接口令牌（Header）
        task_id: 任务ID

    Returns:
        dict: 删除结果
    """
    try:
        success = cluster_manager.delete_task(task_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"任务不存在: {task_id}"
            )

        return {"success": True, "msg": f"任务已删除: {task_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除聚类任务异常: {e}")
        raise HTTPException(
            status_code=500, detail=f"删除聚类任务失败: {str(e)}"
        )
