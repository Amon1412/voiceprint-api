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
        # 启动后台定期清理线程（每小时清理一次过期任务，防止内存泄漏）
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True
        )
        self._cleanup_thread.start()
        logger.info("聚类任务管理器初始化完成")

    def _periodic_cleanup(self) -> None:
        """后台定期清理过期任务"""
        while True:
            time.sleep(3600)  # 每小时执行一次
            try:
                self._cleanup_stale_tasks()
            except Exception as e:
                logger.warning(f"定期清理聚类任务失败: {e}")

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

    def create_task_from_uploads(
        self,
        audio_items: List[Tuple[str, bytes]],
        existing_speaker_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        session_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[str, int, int, List[dict]]:
        """
        从上传的音频数据创建增量聚类任务（支持已有质心作为锚点）

        Args:
            audio_items: [(audio_id, audio_bytes), ...] 音频数据列表
            existing_speaker_ids: 已有说话人ID列表（其质心将作为锚点参与聚类）
            similarity_threshold: 相似度阈值
            session_groups: session分组映射 {session_id: [audio_id, ...]}，
                            提供时启用session级预聚合以提升短音频聚类准确性

        Returns:
            (task_id, total_files, valid_files, invalid_files_list)
        """
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.cluster_similarity_threshold
        )
        task_id = str(uuid.uuid4())

        # 验证音频数据
        valid_items = []
        invalid_files = []
        for audio_id, audio_bytes in audio_items:
            if len(audio_bytes) < 1000:
                invalid_files.append({"file_path": audio_id, "reason": "音频数据过小"})
            else:
                valid_items.append((audio_id, audio_bytes))

        if len(valid_items) < 2:
            raise ValueError(
                f"有效音频不足，至少需要2个，当前只有{len(valid_items)}个"
            )

        # 加载已有说话人的质心和计数
        existing_centroids = {}
        if existing_speaker_ids:
            existing_centroids = voiceprint_db.get_voiceprints_with_count(existing_speaker_ids)
            logger.info(f"加载已有声纹质心: {len(existing_centroids)}个")

        # 创建任务状态
        task = {
            "task_id": task_id,
            "status": "processing",
            "mode": "upload",  # 标记为上传模式
            "audio_items": valid_items,  # [(audio_id, bytes), ...]
            "audio_ids": [item[0] for item in valid_items],
            "total_files": len(valid_items),
            "processed_files": 0,
            "progress_percent": 0.0,
            "similarity_threshold": threshold,
            "existing_centroids": existing_centroids,  # {speaker_id: {"vector": ndarray, "count": int}}
            "session_groups": session_groups,  # {session_id: [audio_id, ...]} or None
            "error": None,
            "result": None,
            "embeddings": {},  # {index: np.ndarray}
            "created_at": datetime.now(),
        }

        with self._lock:
            self._tasks[task_id] = task

        self._cleanup_stale_tasks()

        # 启动后台处理
        thread = threading.Thread(
            target=self._process_upload_task, args=(task_id,), daemon=True
        )
        thread.start()

        logger.info(
            f"增量聚类任务已创建: {task_id}, 有效音频: {len(valid_items)}, "
            f"已有锚点: {len(existing_centroids)}, 阈值: {threshold}, "
            f"session分组: {len(session_groups) if session_groups else 0}个"
        )

        return task_id, len(audio_items), len(valid_items), invalid_files

    def _process_upload_task(self, task_id: str) -> None:
        """后台处理上传模式的聚类任务"""
        start_time = time.time()

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

        audio_items = task["audio_items"]
        audio_ids = task["audio_ids"]
        threshold = task["similarity_threshold"]
        existing_centroids = task["existing_centroids"]
        session_groups = task.get("session_groups")
        embeddings = {}
        valid_indices = []

        logger.info(f"开始处理增量聚类任务: {task_id}, 音频数: {len(audio_items)}")

        try:
            # 逐个提取声纹向量
            for i, (audio_id, audio_bytes) in enumerate(audio_items):
                audio_path = None
                try:
                    audio_path = audio_processor.ensure_16k_wav(audio_bytes)
                    emb = voiceprint_service.extract_voiceprint(audio_path)
                    embeddings[i] = emb
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"提取声纹失败，跳过: {audio_id}, 错误: {e}")
                finally:
                    if audio_path:
                        audio_processor.cleanup_temp_file(audio_path)
                    with self._lock:
                        task["processed_files"] = i + 1
                        task["progress_percent"] = round(
                            (i + 1) / len(audio_items) * 100, 1
                        )

            if len(valid_indices) < 2:
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = (
                        f"有效音频不足，无法聚类（至少需要2个，"
                        f"当前只有{len(valid_indices)}个）"
                    )
                return

            # 判断是否使用session预聚合模式
            use_session_agg = (
                session_groups is not None
                and len(session_groups) >= 2
            )

            if use_session_agg:
                logger.info(
                    f"使用session预聚合模式: {task_id}, "
                    f"session数: {len(session_groups)}, 新向量: {len(valid_indices)}, "
                    f"已有锚点: {len(existing_centroids)}"
                )
                result = self._cluster_with_session_aggregation(
                    embeddings, valid_indices, audio_ids, threshold,
                    existing_centroids, session_groups
                )
            else:
                # 回退到普通聚类
                logger.info(
                    f"开始增量聚类计算: {task_id}, 新向量: {len(valid_indices)}, "
                    f"已有锚点: {len(existing_centroids)}"
                )
                result = self._cluster_with_anchors(
                    embeddings, valid_indices, audio_ids, threshold, existing_centroids
                )

            processing_time = time.time() - start_time
            result["stats"]["processing_time_seconds"] = round(processing_time, 1)

            with self._lock:
                task["status"] = "completed"
                task["result"] = result
                task["embeddings"] = embeddings
                task["progress_percent"] = 100.0
                # 释放音频原始数据，节省内存
                task["audio_items"] = None

            logger.info(
                f"增量聚类任务完成: {task_id}, "
                f"聚类数: {result['stats']['total_clusters']}, "
                f"离群点: {result['stats']['outlier_count']}, "
                f"耗时: {processing_time:.1f}秒"
            )

        except Exception as e:
            with self._lock:
                task["status"] = "failed"
                task["error"] = f"聚类处理异常: {str(e)}"
            logger.error(f"增量聚类任务异常: {task_id}, 错误: {e}")

    def _cluster_with_session_aggregation(
        self,
        embeddings: Dict[int, np.ndarray],
        valid_indices: List[int],
        audio_ids: List[str],
        threshold: float,
        existing_centroids: Dict[str, dict],
        session_groups: Dict[str, List[str]],
    ) -> dict:
        """
        Session级预聚合聚类算法

        核心思路：
        1. 将同一session内的音频embedding取平均，生成session级超级样本
        2. 在session级进行层次聚类（向量更稳定，噪声更小）
        3. 将聚类结果展开回individual audio ID

        短音频（如"嗯嗯"、"我"等1-3秒片段）单独聚类时embedding不稳定，
        但同一session内通常是同一说话人，平均后可大幅降噪。

        Args:
            embeddings: {index: embedding_vector}
            valid_indices: 有效音频索引
            audio_ids: 音频ID列表
            threshold: 相似度阈值
            existing_centroids: {speaker_id: {"vector": ndarray, "count": int}}
            session_groups: {session_id: [audio_id, ...]}
        """
        # 构建 audio_id → index 映射
        aid_to_idx = {}
        for i in valid_indices:
            aid_to_idx[audio_ids[i]] = i

        # Phase 1: 计算每个session的平均embedding
        session_embeddings = {}  # {session_id: np.ndarray}
        session_audio_map = {}   # {session_id: [audio_id, ...]} 仅包含有有效embedding的
        orphan_indices = []      # 不属于任何session的audio index

        # 收集所有session内有效的audio，决定是否聚合
        session_assigned = set()  # 被session聚合吸收的audio_id
        for session_id, session_aids in session_groups.items():
            valid_embs = []
            valid_aids = []
            for aid in session_aids:
                if aid in aid_to_idx:
                    idx = aid_to_idx[aid]
                    if idx in embeddings:
                        valid_embs.append(embeddings[idx])
                        valid_aids.append(aid)

            if len(valid_embs) >= 2:
                # 多条音频的session：计算平均embedding降噪
                session_centroid = self._compute_centroid(
                    [e / max(np.linalg.norm(e), 1e-10) for e in valid_embs]
                )
                session_embeddings[session_id] = session_centroid
                session_audio_map[session_id] = valid_aids
                session_assigned.update(valid_aids)
            # 仅1条或0条音频的session不聚合，让其作为孤立点参与聚类
            # （单条音频没有平均降噪收益，保留细粒度信息）

        # 找出不在任何session中的孤立音频
        for i in valid_indices:
            aid = audio_ids[i]
            if aid not in session_assigned:
                orphan_indices.append(i)

        # 如果session聚合后有效点太少，回退到普通聚类
        total_cluster_points = len(session_embeddings) + len(orphan_indices)
        if total_cluster_points < 2:
            logger.warning(
                f"Session预聚合后有效点不足({total_cluster_points})，回退到普通聚类"
            )
            return self._cluster_with_anchors(
                embeddings, valid_indices, audio_ids, threshold, existing_centroids
            )

        logger.info(
            f"Session预聚合: {len(session_groups)}个session → "
            f"{len(session_embeddings)}个有效session向量 + "
            f"{len(orphan_indices)}个孤立音频, "
            f"总聚类点: {total_cluster_points}"
        )

        # Phase 2: 构建聚类矩阵 = session向量 + 孤立音频向量 + 锚点向量
        session_ids_ordered = list(session_embeddings.keys())
        session_vectors = [session_embeddings[sid] for sid in session_ids_ordered]

        orphan_vectors = []
        for oi in orphan_indices:
            emb = embeddings[oi]
            norm = np.linalg.norm(emb)
            orphan_vectors.append(emb / max(norm, 1e-10))

        anchor_speaker_ids = list(existing_centroids.keys())
        anchor_vectors = [existing_centroids[sid]["vector"] for sid in anchor_speaker_ids]

        all_vectors = session_vectors + orphan_vectors + anchor_vectors
        matrix = np.array(all_vectors, dtype=np.float32)

        n_sessions = len(session_vectors)
        n_orphans = len(orphan_vectors)
        n_anchors = len(anchor_vectors)

        # L2 归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        matrix_normalized = matrix / norms

        # 余弦距离矩阵
        similarity_matrix = np.dot(matrix_normalized, matrix_normalized.T)
        np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.maximum(distance_matrix, 0.0)

        condensed_dist = squareform(distance_matrix)

        # 层次聚类
        linkage_matrix = linkage(condensed_dist, method="average")
        distance_threshold = 1.0 - threshold
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

        # 按标签分组
        cluster_groups_map: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in cluster_groups_map:
                cluster_groups_map[label] = []
            cluster_groups_map[label].append(idx)

        # Phase 3: 构建初始簇 + 离群值挽救
        n_new = n_sessions + n_orphans  # 新数据点数（不含锚点）
        initial_clusters = []
        candidate_outliers = []

        for label in sorted(cluster_groups_map.keys()):
            group_members = cluster_groups_map[label]
            new_members = [m for m in group_members if m < n_new]
            anchor_members = [m for m in group_members if m >= n_new]

            if len(new_members) == 0:
                continue

            if len(new_members) == 1 and len(anchor_members) == 0:
                candidate_outliers.append(new_members[0])
                continue

            initial_clusters.append((label, new_members, anchor_members))

        # 离群值挽救
        rescue_ratio = settings.cluster_outlier_rescue_ratio
        rescue_threshold = 1.0 - (threshold * rescue_ratio)
        final_outliers = []

        if (initial_clusters or anchor_speaker_ids) and candidate_outliers:
            cluster_centroids = []
            for _, new_mems, anchor_mems in initial_clusters:
                all_mems = new_mems + anchor_mems
                group_embs = [matrix_normalized[m] for m in all_mems]
                centroid = self._compute_centroid(group_embs)
                cluster_centroids.append(centroid)

            for outlier_idx in candidate_outliers:
                outlier_vec = matrix_normalized[outlier_idx]
                min_dist = float("inf")
                best_cluster_pos = -1

                for ci, centroid in enumerate(cluster_centroids):
                    sim = float(np.dot(outlier_vec, centroid))
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_pos = ci

                if min_dist <= rescue_threshold and best_cluster_pos >= 0:
                    _, new_mems, anchor_mems = initial_clusters[best_cluster_pos]
                    new_mems.append(outlier_idx)
                else:
                    final_outliers.append(outlier_idx)
        else:
            final_outliers = candidate_outliers

        # Phase 4: 展开session级结果回individual audio IDs
        clusters = []
        outliers = []
        cluster_id_counter = 0

        for _, new_members, anchor_members in initial_clusters:
            # 展开：将session级成员展开为audio_id列表
            expanded_audio_ids = []
            for m in new_members:
                if m < n_sessions:
                    # 这是一个session聚合点，展开为该session的所有audio_id
                    sid = session_ids_ordered[m]
                    expanded_audio_ids.extend(session_audio_map[sid])
                else:
                    # 这是一个孤立音频点
                    orphan_pos = m - n_sessions
                    orphan_original_idx = orphan_indices[orphan_pos]
                    expanded_audio_ids.append(audio_ids[orphan_original_idx])

            # 计算该聚类所有individual embedding的质心（用于计算距离）
            cluster_individual_embs = []
            for aid in expanded_audio_ids:
                if aid in aid_to_idx and aid_to_idx[aid] in embeddings:
                    emb = embeddings[aid_to_idx[aid]]
                    norm = np.linalg.norm(emb)
                    cluster_individual_embs.append(emb / max(norm, 1e-10))

            if not cluster_individual_embs:
                continue

            centroid = self._compute_centroid(cluster_individual_embs)

            # 构建files_info，计算每个audio到质心的距离
            files_info = []
            for aid in expanded_audio_ids:
                if aid in aid_to_idx and aid_to_idx[aid] in embeddings:
                    emb = embeddings[aid_to_idx[aid]]
                    norm = np.linalg.norm(emb)
                    emb_normalized = emb / max(norm, 1e-10)
                    sim = float(np.dot(emb_normalized, centroid))
                    distance = round(1.0 - sim, 4)
                    files_info.append({"audio_id": aid, "distance_to_centroid": distance})

            files_info.sort(key=lambda x: x["distance_to_centroid"])

            cluster_info = {
                "cluster_id": cluster_id_counter,
                "file_count": len(files_info),
                "files": files_info,
                "existing_speaker_id": None,
                "existing_speaker_count": None,
            }

            # 检查是否匹配已有说话人锚点
            if anchor_members:
                anchor_idx = anchor_members[0] - n_new
                matched_speaker_id = anchor_speaker_ids[anchor_idx]
                matched_count = existing_centroids[matched_speaker_id]["count"]
                cluster_info["existing_speaker_id"] = matched_speaker_id
                cluster_info["existing_speaker_count"] = matched_count

            clusters.append(cluster_info)
            cluster_id_counter += 1

        # 展开离群session/孤立音频
        for outlier_idx in final_outliers:
            if outlier_idx < n_sessions:
                sid = session_ids_ordered[outlier_idx]
                for aid in session_audio_map[sid]:
                    outliers.append(aid)
            else:
                orphan_pos = outlier_idx - n_sessions
                orphan_original_idx = orphan_indices[orphan_pos]
                outliers.append(audio_ids[orphan_original_idx])

        rescued_count = len(candidate_outliers) - len(final_outliers)
        if rescued_count > 0:
            logger.info(
                f"Session聚类离群值挽救: {rescued_count}/{len(candidate_outliers)} 个被归入现有簇"
            )

        total_audio_in_clusters = sum(c["file_count"] for c in clusters)
        logger.info(
            f"Session预聚合聚类完成: {len(clusters)}个簇 "
            f"(含{total_audio_in_clusters}条音频), "
            f"{len(outliers)}个离群音频"
        )

        return {
            "clusters": clusters,
            "outliers": outliers,
            "stats": {
                "total_files": len(audio_ids),
                "valid_files": len(valid_indices),
                "total_clusters": len(clusters),
                "outlier_count": len(outliers),
                "rescued_outliers": rescued_count,
                "session_count": len(session_embeddings),
                "orphan_count": len(orphan_indices),
                "processing_time_seconds": 0.0,
            },
        }

    def _cluster_with_anchors(
        self,
        embeddings: Dict[int, np.ndarray],
        valid_indices: List[int],
        audio_ids: List[str],
        threshold: float,
        existing_centroids: Dict[str, dict],
    ) -> dict:
        """
        带已有质心锚点的聚类算法

        新音频向量和已有质心一起参与层次聚类。
        已有质心作为锚点，聚类后检查每个分组是否包含锚点：
        - 包含锚点 → 标记为已有说话人
        - 不包含锚点 → 标记为新说话人

        Args:
            embeddings: {index: embedding_vector}
            valid_indices: 有效音频索引
            audio_ids: 音频ID列表
            threshold: 相似度阈值
            existing_centroids: {speaker_id: {"vector": ndarray, "count": int}}
        """
        indices = sorted(valid_indices)
        new_vectors = [embeddings[i] for i in indices]

        # 合并矩阵：新音频向量 + 已有质心
        anchor_speaker_ids = list(existing_centroids.keys())
        anchor_vectors = [existing_centroids[sid]["vector"] for sid in anchor_speaker_ids]

        all_vectors = new_vectors + anchor_vectors
        matrix = np.array(all_vectors, dtype=np.float32)

        n_new = len(new_vectors)
        n_anchors = len(anchor_vectors)
        total = n_new + n_anchors

        # L2 归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        matrix_normalized = matrix / norms

        # 余弦相似度 → 距离矩阵
        similarity_matrix = np.dot(matrix_normalized, matrix_normalized.T)
        np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.maximum(distance_matrix, 0.0)

        condensed_dist = squareform(distance_matrix)

        # 层次聚类
        linkage_matrix = linkage(condensed_dist, method="average")
        distance_threshold = 1.0 - threshold
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

        # 按标签分组
        cluster_groups: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(idx)

        # 第一轮：构建初始簇和候选离群值
        initial_clusters = []  # [(label, [new_members], [anchor_members])]
        candidate_outliers = []  # [(matrix_index, original_audio_index)]

        for label in sorted(cluster_groups.keys()):
            group_members = cluster_groups[label]
            new_members = [m for m in group_members if m < n_new]
            anchor_members = [m for m in group_members if m >= n_new]

            if len(new_members) == 0:
                continue

            if len(new_members) == 1 and len(anchor_members) == 0:
                candidate_outliers.append(new_members[0])
                continue

            initial_clusters.append((label, new_members, anchor_members))

        # 第二轮：离群值挽救 — 尝试将单点归入最近的簇或锚点
        rescue_ratio = settings.cluster_outlier_rescue_ratio
        rescue_threshold = 1.0 - (threshold * rescue_ratio)
        final_outliers = []

        if (initial_clusters or anchor_speaker_ids) and candidate_outliers:
            # 计算每个簇（含锚点）的质心
            cluster_centroids = []
            for _, new_mems, anchor_mems in initial_clusters:
                all_mems = new_mems + anchor_mems
                group_embs = [matrix_normalized[m] for m in all_mems]
                centroid = self._compute_centroid(group_embs)
                cluster_centroids.append(centroid)

            for outlier_idx in candidate_outliers:
                outlier_vec = matrix_normalized[outlier_idx]
                min_dist = float("inf")
                best_cluster_pos = -1

                for ci, centroid in enumerate(cluster_centroids):
                    sim = float(np.dot(outlier_vec, centroid))
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_pos = ci

                if min_dist <= rescue_threshold and best_cluster_pos >= 0:
                    _, new_mems, anchor_mems = initial_clusters[best_cluster_pos]
                    new_mems.append(outlier_idx)
                    logger.debug(
                        f"锚点聚类离群值已挽救: index={outlier_idx}, "
                        f"归入簇{best_cluster_pos}, 距离={min_dist:.4f}"
                    )
                else:
                    final_outliers.append(outlier_idx)
        else:
            final_outliers = candidate_outliers

        # 第三轮：构建最终结果
        clusters = []
        outliers = []
        cluster_id_counter = 0

        for _, new_members, anchor_members in initial_clusters:
            group_new_embeddings = [matrix_normalized[m] for m in new_members]
            centroid = self._compute_centroid(group_new_embeddings)

            files_info = []
            for m in new_members:
                original_idx = indices[m]
                aid = audio_ids[original_idx]
                sim = float(np.dot(matrix_normalized[m], centroid))
                distance = round(1.0 - sim, 4)
                files_info.append({"audio_id": aid, "distance_to_centroid": distance})

            files_info.sort(key=lambda x: x["distance_to_centroid"])

            cluster_info = {
                "cluster_id": cluster_id_counter,
                "file_count": len(new_members),
                "files": files_info,
                "existing_speaker_id": None,
                "existing_speaker_count": None,
            }

            if anchor_members:
                anchor_idx = anchor_members[0] - n_new
                matched_speaker_id = anchor_speaker_ids[anchor_idx]
                matched_count = existing_centroids[matched_speaker_id]["count"]
                cluster_info["existing_speaker_id"] = matched_speaker_id
                cluster_info["existing_speaker_count"] = matched_count

            clusters.append(cluster_info)
            cluster_id_counter += 1

        for outlier_idx in final_outliers:
            original_idx = indices[outlier_idx]
            outliers.append(audio_ids[original_idx])

        rescued_count = len(candidate_outliers) - len(final_outliers)
        if rescued_count > 0:
            logger.info(
                f"锚点聚类离群值挽救完成: {rescued_count}/{len(candidate_outliers)} 个离群点被归入现有簇"
            )

        return {
            "clusters": clusters,
            "outliers": outliers,
            "stats": {
                "total_files": len(audio_ids),
                "valid_files": len(valid_indices),
                "total_clusters": len(clusters),
                "outlier_count": len(outliers),
                "rescued_outliers": rescued_count,
                "processing_time_seconds": 0.0,
            },
        }

    def confirm_clusters_merge(
        self, task_id: str, assignments: List[dict]
    ) -> dict:
        """
        确认增量聚类结果并注册/合并声纹

        Args:
            task_id: 任务ID
            assignments: [{"cluster_id": int, "speaker_id": str, "merge_with_existing": bool}, ...]

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
        audio_ids = task["audio_ids"]
        existing_centroids = task.get("existing_centroids", {})
        indices = sorted([i for i in embeddings.keys()])

        registered = []
        failed = []

        for assignment in assignments:
            cluster_id = assignment["cluster_id"]
            speaker_id = assignment["speaker_id"]
            merge_with_existing = assignment.get("merge_with_existing", False)

            if cluster_id not in cluster_map:
                failed.append({
                    "speaker_id": speaker_id,
                    "cluster_id": cluster_id,
                    "reason": f"聚类ID不存在: {cluster_id}",
                })
                continue

            try:
                cluster = cluster_map[cluster_id]

                # 收集该聚类中新音频的向量
                cluster_embeddings = []
                for file_info in cluster["files"]:
                    aid = file_info["audio_id"]
                    try:
                        idx = audio_ids.index(aid)
                    except ValueError:
                        logger.warning(f"audio_id {aid} 不在原始列表中，跳过")
                        continue
                    if idx in embeddings:
                        cluster_embeddings.append(embeddings[idx])

                if not cluster_embeddings:
                    failed.append({
                        "speaker_id": speaker_id,
                        "cluster_id": cluster_id,
                        "reason": "聚类中无有效向量",
                    })
                    continue

                # 归一化
                normalized = []
                for emb in cluster_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-10:
                        normalized.append(emb / norm)
                    else:
                        normalized.append(emb)

                new_count = len(normalized)

                if merge_with_existing and speaker_id in existing_centroids:
                    # 加权合并已有质心
                    old_data = existing_centroids[speaker_id]
                    old_centroid = old_data["vector"]
                    old_count = old_data["count"]

                    # 归一化已有质心
                    old_norm = np.linalg.norm(old_centroid)
                    if old_norm > 1e-10:
                        old_centroid = old_centroid / old_norm

                    # 新音频的加权和
                    new_sum = np.sum(normalized, axis=0)

                    # 加权合并: merged = (old * old_count + new_sum) / (old_count + new_count)
                    merged = (old_centroid * old_count + new_sum) / (old_count + new_count)

                    # L2 归一化
                    merged_norm = np.linalg.norm(merged)
                    if merged_norm > 1e-10:
                        merged = merged / merged_norm
                    merged = merged.astype(np.float32)

                    total_count = old_count + new_count

                    success = voiceprint_db.update_voiceprint_merge(
                        speaker_id, merged, total_count
                    )
                    if success:
                        registered.append({
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "file_count": new_count,
                            "merged": True,
                            "total_count": total_count,
                        })
                        logger.info(
                            f"聚类质心合并成功: {speaker_id}, "
                            f"新增{new_count}个, 总计{total_count}个"
                        )
                    else:
                        failed.append({
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "reason": "数据库合并更新失败",
                        })
                else:
                    # 新建声纹
                    centroid = self._compute_centroid(normalized)
                    success = voiceprint_db.save_voiceprint_with_count(
                        speaker_id, centroid, new_count
                    )
                    if success:
                        registered.append({
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "file_count": new_count,
                            "merged": False,
                            "total_count": new_count,
                        })
                        logger.info(
                            f"聚类声纹新建成功: {speaker_id}, 音频数: {new_count}"
                        )
                    else:
                        failed.append({
                            "speaker_id": speaker_id,
                            "cluster_id": cluster_id,
                            "reason": "数据库保存失败",
                        })

            except Exception as e:
                logger.error(
                    f"增量聚类注册异常: {speaker_id}, 聚类ID: {cluster_id}, 错误: {e}"
                )
                failed.append({
                    "speaker_id": speaker_id,
                    "cluster_id": cluster_id,
                    "reason": str(e),
                })

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

        # 第一轮：构建初始簇和候选离群值
        initial_clusters = []  # [(label, [indices_in_matrix])]
        candidate_outliers = []  # [index_in_matrix]

        for label in sorted(cluster_groups.keys()):
            group_indices_in_matrix = cluster_groups[label]
            if len(group_indices_in_matrix) == 1:
                candidate_outliers.append(group_indices_in_matrix[0])
            else:
                initial_clusters.append((label, group_indices_in_matrix))

        # 第二轮：离群值挽救 — 尝试将单点归入最近的簇
        rescue_ratio = settings.cluster_outlier_rescue_ratio
        rescue_threshold = 1.0 - (threshold * rescue_ratio)  # 更宽松的距离阈值
        final_outliers = []

        if initial_clusters and candidate_outliers:
            # 计算每个初始簇的质心
            cluster_centroids = []
            for _, group_idx in initial_clusters:
                group_embs = [matrix_normalized[i] for i in group_idx]
                centroid = self._compute_centroid(group_embs)
                cluster_centroids.append(centroid)

            for outlier_idx in candidate_outliers:
                outlier_vec = matrix_normalized[outlier_idx]
                # 计算到每个簇质心的距离
                min_dist = float("inf")
                best_cluster_pos = -1
                for ci, centroid in enumerate(cluster_centroids):
                    sim = float(np.dot(outlier_vec, centroid))
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_pos = ci

                if min_dist <= rescue_threshold and best_cluster_pos >= 0:
                    # 归入最近的簇
                    _, group_idx = initial_clusters[best_cluster_pos]
                    group_idx.append(outlier_idx)
                    logger.debug(
                        f"离群值已挽救: index={outlier_idx}, "
                        f"归入簇{best_cluster_pos}, 距离={min_dist:.4f}"
                    )
                else:
                    final_outliers.append(outlier_idx)
        else:
            final_outliers = candidate_outliers

        # 第三轮：构建最终结果
        clusters = []
        outliers = []
        cluster_id_counter = 0

        for _, group_indices_in_matrix in initial_clusters:
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

        for outlier_idx in final_outliers:
            original_idx = indices[outlier_idx]
            outliers.append({"file_path": file_paths[original_idx]})

        rescued_count = len(candidate_outliers) - len(final_outliers)
        if rescued_count > 0:
            logger.info(
                f"离群值挽救完成: {rescued_count}/{len(candidate_outliers)} 个离群点被归入现有簇"
            )

        return {
            "clusters": clusters,
            "outliers": outliers,
            "stats": {
                "total_files": len(file_paths),
                "valid_files": len(valid_indices),
                "total_clusters": len(clusters),
                "outlier_count": len(outliers),
                "rescued_outliers": rescued_count,
                "processing_time_seconds": 0.0,
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
