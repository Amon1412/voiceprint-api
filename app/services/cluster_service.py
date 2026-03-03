import os
import uuid
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ..core.config import settings
from ..core.logger import get_logger
from .voiceprint_service import voiceprint_service
from ..database.voiceprint_db import voiceprint_db
from ..utils.audio_utils import audio_processor

logger = get_logger(__name__)


class ClusterTaskManager:
    """声纹聚类任务管理器"""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()
        logger.info("聚类任务管理器初始化完成")

    def create_task(
        self,
        file_paths: List[str],
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[str, int, int, List[dict]]:
        """
        创建聚类任务

        Args:
            file_paths: 音频文件路径列表
            similarity_threshold: 相似度阈值，None则使用配置默认值

        Returns:
            (task_id, total_files, valid_files, invalid_files_list)
        """
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.cluster_similarity_threshold
        )
        task_id = str(uuid.uuid4())

        # 验证文件
        valid_paths = []
        invalid_files = []
        for fp in file_paths:
            if not os.path.isfile(fp):
                invalid_files.append({"file_path": fp, "reason": "文件不存在"})
            elif not fp.lower().endswith(".wav"):
                invalid_files.append(
                    {"file_path": fp, "reason": "不是WAV格式文件"}
                )
            else:
                valid_paths.append(fp)

        if len(valid_paths) < 2:
            raise ValueError(
                f"有效音频文件不足，至少需要2个，当前只有{len(valid_paths)}个"
            )

        # 创建任务状态
        task = {
            "task_id": task_id,
            "status": "processing",
            "file_paths": valid_paths,
            "total_files": len(valid_paths),
            "processed_files": 0,
            "progress_percent": 0.0,
            "similarity_threshold": threshold,
            "error": None,
            "result": None,
            "embeddings": {},  # {index: np.ndarray}
            "created_at": datetime.now(),
        }

        with self._lock:
            self._tasks[task_id] = task

        # 清理过期任务
        self._cleanup_stale_tasks()

        # 启动后台处理线程
        thread = threading.Thread(
            target=self._process_task, args=(task_id,), daemon=True
        )
        thread.start()

        logger.info(
            f"聚类任务已创建: {task_id}, 有效文件: {len(valid_paths)}, "
            f"无效文件: {len(invalid_files)}, 阈值: {threshold}"
        )

        return task_id, len(file_paths), len(valid_paths), invalid_files

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务状态和结果"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None

            # 构建返回数据（不暴露内部向量数据）
            response = {
                "task_id": task["task_id"],
                "status": task["status"],
                "total_files": task["total_files"],
                "processed_files": task["processed_files"],
                "progress_percent": task["progress_percent"],
                "error": task["error"],
            }

            if task["status"] == "completed" and task["result"] is not None:
                response["clusters"] = task["result"]["clusters"]
                response["outliers"] = task["result"]["outliers"]
                response["stats"] = task["result"]["stats"]

            return response

    def confirm_clusters(
        self, task_id: str, assignments: List[dict]
    ) -> dict:
        """
        确认聚类结果并注册声纹

        Args:
            task_id: 任务ID
            assignments: [{"cluster_id": int, "speaker_id": str}, ...]

        Returns:
            {"success": bool, "registered": [...], "failed": [...]}
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"任务不存在: {task_id}")
            if task["status"] != "completed":
                raise RuntimeError(
                    f"任务状态不允许确认操作，当前状态: {task['status']}"
                )

        result = task["result"]
        cluster_map = {c["cluster_id"]: c for c in result["clusters"]}
        embeddings = task["embeddings"]
        file_paths = task["file_paths"]

        registered = []
        failed = []

        for assignment in assignments:
            cluster_id = assignment["cluster_id"]
            speaker_id = assignment["speaker_id"]

            if cluster_id not in cluster_map:
                failed.append(
                    {
                        "speaker_id": speaker_id,
                        "cluster_id": cluster_id,
                        "reason": f"聚类ID不存在: {cluster_id}",
                    }
                )
                continue

            try:
                cluster = cluster_map[cluster_id]
                # 收集该聚类所有文件的向量
                cluster_embeddings = []
                for file_info in cluster["files"]:
                    fp = file_info["file_path"]
                    idx = file_paths.index(fp)
                    if idx in embeddings:
                        cluster_embeddings.append(embeddings[idx])

                if not cluster_embeddings:
                    failed.append(
                        {
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "reason": "聚类中无有效向量",
                        }
                    )
                    continue

                # 归一化后计算质心（与聚类阶段保持一致）
                normalized = []
                for emb in cluster_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-10:
                        normalized.append(emb / norm)
                    else:
                        normalized.append(emb)
                centroid = self._compute_centroid(normalized)

                # 注册到数据库
                success = voiceprint_db.save_voiceprint(speaker_id, centroid)
                if success:
                    registered.append(
                        {
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "file_count": len(cluster_embeddings),
                        }
                    )
                    logger.info(
                        f"聚类声纹注册成功: {speaker_id}, "
                        f"聚类ID: {cluster_id}, 文件数: {len(cluster_embeddings)}"
                    )
                else:
                    failed.append(
                        {
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "reason": "数据库保存失败",
                        }
                    )

            except Exception as e:
                logger.error(
                    f"聚类声纹注册异常: {speaker_id}, 聚类ID: {cluster_id}, 错误: {e}"
                )
                failed.append(
                    {
                        "speaker_id": speaker_id,
                        "cluster_id": cluster_id,
                        "reason": str(e),
                    }
                )

        # 更新任务状态
        with self._lock:
            task["status"] = "confirmed"

        return {
            "success": len(failed) == 0,
            "registered": registered,
            "failed": failed,
        }

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"聚类任务已删除: {task_id}")
                return True
            return False

    def _process_task(self, task_id: str) -> None:
        """后台处理任务"""
        start_time = time.time()

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

        file_paths = task["file_paths"]
        threshold = task["similarity_threshold"]
        embeddings = {}
        valid_indices = []

        logger.info(
            f"开始处理聚类任务: {task_id}, 文件数: {len(file_paths)}"
        )

        try:
            # 逐文件提取声纹向量
            for i, fp in enumerate(file_paths):
                audio_path = None
                try:
                    # 读取文件并转换为16k WAV
                    with open(fp, "rb") as f:
                        audio_bytes = f.read()

                    if len(audio_bytes) < 1000:
                        logger.warning(f"音频文件过小，跳过: {fp}")
                        continue

                    audio_path = audio_processor.ensure_16k_wav(audio_bytes)
                    emb = voiceprint_service.extract_voiceprint(audio_path)
                    embeddings[i] = emb
                    valid_indices.append(i)

                except Exception as e:
                    logger.warning(f"提取声纹失败，跳过: {fp}, 错误: {e}")
                finally:
                    if audio_path:
                        audio_processor.cleanup_temp_file(audio_path)
                    # 更新进度（放在finally确保continue不会跳过）
                    with self._lock:
                        task["processed_files"] = i + 1
                        task["progress_percent"] = round(
                            (i + 1) / len(file_paths) * 100, 1
                        )

            # 检查有效向量数
            if len(valid_indices) < 2:
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = (
                        f"有效音频不足，无法聚类（至少需要2个，"
                        f"当前只有{len(valid_indices)}个）"
                    )
                logger.error(
                    f"聚类任务失败: {task_id}, 有效向量不足: {len(valid_indices)}"
                )
                return

            # 执行聚类
            logger.info(
                f"开始聚类计算: {task_id}, 有效向量: {len(valid_indices)}"
            )
            result = self._cluster_embeddings(
                embeddings, valid_indices, file_paths, threshold
            )

            processing_time = time.time() - start_time
            result["stats"]["processing_time_seconds"] = round(
                processing_time, 1
            )

            # 更新任务
            with self._lock:
                task["status"] = "completed"
                task["result"] = result
                task["embeddings"] = embeddings
                task["progress_percent"] = 100.0

            logger.info(
                f"聚类任务完成: {task_id}, "
                f"聚类数: {result['stats']['total_clusters']}, "
                f"离群点: {result['stats']['outlier_count']}, "
                f"耗时: {processing_time:.1f}秒"
            )

        except Exception as e:
            with self._lock:
                task["status"] = "failed"
                task["error"] = f"聚类处理异常: {str(e)}"
            logger.error(f"聚类任务异常: {task_id}, 错误: {e}")

    def _cluster_embeddings(
        self,
        embeddings: Dict[int, np.ndarray],
        valid_indices: List[int],
        file_paths: List[str],
        threshold: float,
    ) -> dict:
        """
        对声纹特征向量进行层次聚类

        Args:
            embeddings: {file_index: embedding_vector}
            valid_indices: 有效文件的索引列表
            file_paths: 完整文件路径列表
            threshold: 相似度阈值

        Returns:
            聚类结果字典
        """
        # 构建向量矩阵
        indices = sorted(valid_indices)
        matrix = np.array([embeddings[i] for i in indices], dtype=np.float32)

        # L2 归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # 避免除零
        matrix_normalized = matrix / norms

        # 余弦相似度矩阵
        similarity_matrix = np.dot(matrix_normalized, matrix_normalized.T)
        np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)

        # 转距离矩阵
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        # 确保非负（浮点精度问题）
        distance_matrix = np.maximum(distance_matrix, 0.0)

        # 转换为压缩形式
        condensed_dist = squareform(distance_matrix)

        # 层次聚类
        linkage_matrix = linkage(condensed_dist, method="average")
        distance_threshold = 1.0 - threshold
        labels = fcluster(
            linkage_matrix, t=distance_threshold, criterion="distance"
        )

        # 按标签分组
        cluster_groups: Dict[int, List[int]] = {}
        for idx_in_matrix, label in enumerate(labels):
            label = int(label)
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(idx_in_matrix)

        # 构建结果
        clusters = []
        outliers = []
        cluster_id_counter = 0

        for label in sorted(cluster_groups.keys()):
            group_indices_in_matrix = cluster_groups[label]

            if len(group_indices_in_matrix) == 1:
                # 单文件聚类 → 离群点
                original_idx = indices[group_indices_in_matrix[0]]
                outliers.append({"file_path": file_paths[original_idx]})
                continue

            # 计算质心
            group_embeddings = [
                matrix_normalized[i] for i in group_indices_in_matrix
            ]
            centroid = self._compute_centroid(group_embeddings)

            # 计算每个文件到质心的距离
            files_info = []
            min_distance = float("inf")
            centroid_file = ""

            for idx_in_matrix in group_indices_in_matrix:
                original_idx = indices[idx_in_matrix]
                fp = file_paths[original_idx]
                # 到质心的余弦距离
                sim = float(
                    np.dot(matrix_normalized[idx_in_matrix], centroid)
                )
                distance = round(1.0 - sim, 4)

                files_info.append(
                    {"file_path": fp, "distance_to_centroid": distance}
                )

                if distance < min_distance:
                    min_distance = distance
                    centroid_file = fp

            # 按距离排序
            files_info.sort(key=lambda x: x["distance_to_centroid"])

            clusters.append(
                {
                    "cluster_id": cluster_id_counter,
                    "file_count": len(group_indices_in_matrix),
                    "centroid_file": centroid_file,
                    "files": files_info,
                }
            )
            cluster_id_counter += 1

        return {
            "clusters": clusters,
            "outliers": outliers,
            "stats": {
                "total_files": len(file_paths),
                "valid_files": len(valid_indices),
                "total_clusters": len(clusters),
                "outlier_count": len(outliers),
                "processing_time_seconds": 0.0,  # 由调用者填充
            },
        }

    def _compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """计算聚类质心（嵌入向量的均值并归一化）"""
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def _cleanup_stale_tasks(self) -> None:
        """清理过期任务"""
        expire_hours = settings.cluster_task_expire_hours
        cutoff = datetime.now() - timedelta(hours=expire_hours)
        stale_ids = []

        with self._lock:
            for task_id, task in self._tasks.items():
                if task["created_at"] < cutoff:
                    stale_ids.append(task_id)

            for task_id in stale_ids:
                del self._tasks[task_id]

        if stale_ids:
            logger.info(f"清理过期聚类任务: {len(stale_ids)}个")


# 全局聚类任务管理器实例
cluster_manager = ClusterTaskManager()
