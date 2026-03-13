import numpy as np
import torch
import time
import os
import threading
from typing import Dict, List, Tuple
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from ..core.config import settings
from ..core.logger import get_logger
from ..database.voiceprint_db import voiceprint_db
from ..utils.audio_utils import audio_processor

logger = get_logger(__name__)


class VoiceprintService:
    """声纹识别服务类"""

    def __init__(self):
        self._pipeline = None
        self.similarity_threshold = settings.similarity_threshold
        self._pipeline_lock = threading.Lock()  # 添加线程锁
        self._init_pipeline()
        self._warmup_model()  # 添加模型预热

    def _init_pipeline(self) -> None:
        """初始化声纹识别模型"""
        start_time = time.time()
        logger.start("初始化声纹识别模型")

        try:
            # 检查CUDA可用性
            if torch.cuda.is_available():
                device = "gpu"
                logger.info(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("使用CPU设备")

            logger.info("开始加载模型: iic/speech_campplus_sv_zh-cn_3dspeaker_16k")
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model="iic/speech_campplus_sv_zh-cn_3dspeaker_16k",
                device=device,
            )

            init_time = time.time() - start_time
            logger.complete("初始化声纹识别模型", init_time)
        except Exception as e:
            init_time = time.time() - start_time
            logger.fail(f"声纹模型加载失败，耗时: {init_time:.3f}秒，错误: {e}")
            raise

    def _warmup_model(self) -> None:
        """模型预热，避免第一次推理的延迟"""
        start_time = time.time()
        logger.start("开始模型预热")

        try:
            # 预热librosa重采样组件
            logger.debug("预热librosa重采样组件...")
            import librosa
            import soundfile as sf
            import tempfile

            # 生成一个更真实的测试音频（1秒的随机音频，模拟真实语音）
            sample_rate = 16000
            duration = 1.0  # 1秒
            samples = int(sample_rate * duration)

            # 生成随机音频数据，模拟真实语音
            np.random.seed(42)  # 固定随机种子，确保可重现
            test_audio = (
                np.random.randn(samples).astype(np.float32) * 0.1
            )  # 小幅度随机音频

            # 创建不同采样率的音频文件进行预热
            test_rates = [8000, 22050, 44100, 16000]  # 测试不同采样率

            for test_rate in test_rates:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                    # 生成测试采样率的音频
                    test_samples = int(test_rate * duration)
                    test_audio_resampled = librosa.resample(
                        test_audio, orig_sr=sample_rate, target_sr=test_rate
                    )
                    sf.write(tmpf.name, test_audio_resampled, test_rate)
                    temp_audio_path = tmpf.name

                try:
                    # 使用音频处理器处理这个文件（预热音频处理流程）
                    with open(temp_audio_path, "rb") as f:
                        audio_bytes = f.read()

                    # 预热音频处理
                    processed_path = audio_processor.ensure_16k_wav(audio_bytes)

                    # 预热模型推理
                    with self._pipeline_lock:
                        result = self._pipeline([processed_path], output_emb=True)
                        emb = self._to_numpy(result["embs"][0]).astype(np.float32)
                        logger.debug(
                            f"模型预热完成 ({test_rate}Hz -> 16kHz)，特征维度: {emb.shape}"
                        )

                    # 清理处理后的文件
                    audio_processor.cleanup_temp_file(processed_path)

                finally:
                    # 清理临时文件
                    import os

                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)

            warmup_time = time.time() - start_time
            logger.complete("模型预热完成", warmup_time)

        except Exception as e:
            warmup_time = time.time() - start_time
            logger.warning(f"模型预热失败，耗时: {warmup_time:.3f}秒，错误: {e}")
            # 预热失败不影响服务启动，只记录警告

    def _to_numpy(self, x) -> np.ndarray:
        """
        将torch tensor或其他类型转为numpy数组

        Args:
            x: 输入数据

        Returns:
            np.ndarray: numpy数组
        """
        return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    @staticmethod
    def _get_rss_mb() -> int:
        """读取当前进程RSS内存(MB)，用于OOM诊断"""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return -1

    def extract_voiceprint(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取声纹特征

        Args:
            audio_path: 音频文件路径

        Returns:
            np.ndarray: 声纹特征向量
        """
        start_time = time.time()
        # 记录音频文件大小和进程内存，用于OOM诊断
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        mem_mb = self._get_rss_mb()
        logger.start(f"提取声纹特征，音频文件: {audio_path}，文件大小: {file_size}字节，进程RSS: {mem_mb}MB")

        try:
            # 使用线程锁确保模型推理的线程安全
            with self._pipeline_lock:
                pipeline_start = time.time()
                logger.debug("开始模型推理...")

                # 检查pipeline是否可用
                if self._pipeline is None:
                    raise RuntimeError("声纹模型未初始化")

                result = self._pipeline([audio_path], output_emb=True)
                pipeline_time = time.time() - pipeline_start
                logger.debug(f"模型推理完成，耗时: {pipeline_time:.3f}秒")

            convert_start = time.time()
            emb = self._to_numpy(result["embs"][0]).astype(np.float32)
            convert_time = time.time() - convert_start
            logger.debug(f"数据转换完成，耗时: {convert_time:.3f}秒")

            total_time = time.time() - start_time
            mem_after = self._get_rss_mb()
            logger.complete(f"提取声纹特征，维度: {emb.shape}，推理后RSS: {mem_after}MB", total_time)
            return emb
        except Exception as e:
            total_time = time.time() - start_time
            logger.fail(f"声纹特征提取失败，总耗时: {total_time:.3f}秒，错误: {e}")
            raise

    def extract_voiceprint_batch(self, audio_paths: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """
        批量提取声纹特征，利用 ModelScope pipeline 的批量推理能力

        Args:
            audio_paths: 音频文件路径列表
            batch_size: 每批推理的音频数量，避免单次占用过多内存

        Returns:
            List[np.ndarray]: 声纹特征向量列表
        """
        if not audio_paths:
            return []

        start_time = time.time()
        all_embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_start = time.time()

            with self._pipeline_lock:
                if self._pipeline is None:
                    raise RuntimeError("声纹模型未初始化")
                result = self._pipeline(batch_paths, output_emb=True)

            batch_embs = [self._to_numpy(emb).astype(np.float32) for emb in result["embs"]]
            all_embeddings.extend(batch_embs)

            batch_time = time.time() - batch_start
            logger.debug(f"批量推理[{i}:{i+len(batch_paths)}] 完成，耗时: {batch_time:.3f}秒")

        total_time = time.time() - start_time
        logger.info(f"批量提取声纹特征完成，共{len(audio_paths)}条，总耗时: {total_time:.3f}秒")
        return all_embeddings

    def calculate_similarity_matrix(self, test_embs: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """
        批量计算相似度矩阵，使用矩阵乘法替代逐对循环

        Args:
            test_embs: [M, D] M条测试音频的embedding
            candidate_embs: [N, D] N个候选声纹的embedding

        Returns:
            np.ndarray: [M, N] 相似度矩阵
        """
        # L2 归一化
        test_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
        cand_norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        test_normalized = test_embs / np.maximum(test_norms, 1e-8)
        cand_normalized = candidate_embs / np.maximum(cand_norms, 1e-8)
        # 矩阵乘法一次计算所有余弦相似度
        return test_normalized @ cand_normalized.T

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个声纹特征的相似度

        Args:
            emb1: 声纹特征1
            emb2: 声纹特征2

        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 使用余弦相似度
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0

    def register_voiceprint(self, speaker_id: str, audio_bytes: bytes) -> bool:
        """
        注册声纹

        Args:
            speaker_id: 说话人ID
            audio_bytes: 音频字节数据

        Returns:
            bool: 注册是否成功
        """
        audio_path = None
        try:
            # 简化音频验证，只做基本检查
            if len(audio_bytes) < 1000:  # 文件太小
                logger.warning(f"音频文件过小: {speaker_id}")
                return False

            # 处理音频文件（注册时不降噪，保留原始声纹特征）
            audio_path = audio_processor.ensure_16k_wav(audio_bytes, apply_denoise=False)

            # 提取声纹特征
            emb = self.extract_voiceprint(audio_path)

            # 保存到数据库
            success = voiceprint_db.save_voiceprint(speaker_id, emb)

            if success:
                logger.info(f"声纹注册成功: {speaker_id}")
            else:
                logger.error(f"声纹注册失败: {speaker_id}")

            return success

        except Exception as e:
            logger.error(f"声纹注册异常 {speaker_id}: {e}")
            return False
        finally:
            # 清理临时文件
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)

    def register_voiceprint_multi(self, speaker_id: str, audio_bytes_list: List[bytes]) -> Tuple[bool, int]:
        """
        从多个音频提取embedding，取均值注册声纹

        Args:
            speaker_id: 说话人ID
            audio_bytes_list: 多个音频字节数据列表

        Returns:
            Tuple[bool, int]: (是否成功, 有效embedding数量)
        """
        embeddings = []
        for i, audio_bytes in enumerate(audio_bytes_list):
            audio_path = None
            try:
                if len(audio_bytes) < 1000:
                    logger.warning(f"音频文件{i}过小，跳过")
                    continue
                audio_path = audio_processor.ensure_16k_wav(audio_bytes, apply_denoise=False)
                emb = self.extract_voiceprint(audio_path)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"处理音频文件{i}失败: {e}")
            finally:
                if audio_path:
                    audio_processor.cleanup_temp_file(audio_path)

        if not embeddings:
            logger.error(f"没有有效的embedding可注册: {speaker_id}")
            return False, 0

        # 取均值并L2正规化
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        success = voiceprint_db.save_voiceprint_with_count(speaker_id, mean_emb, len(embeddings))
        if success:
            logger.info(f"多文件声纹注册成功: {speaker_id}, 使用{len(embeddings)}个embedding")
        return success, len(embeddings)

    def _check_audio_quality(self, audio_bytes: bytes) -> Dict:
        """
        对原始音频字节进行质量预检（内存读取，不写磁盘）

        Args:
            audio_bytes: 原始音频字节

        Returns:
            dict: 质量指标
        """
        import io
        import soundfile as sf
        try:
            buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf)
            return audio_processor.check_audio_quality(data, sr)
        except Exception as e:
            logger.warning(f"音频质量预检失败: {e}")
            return {"ok": True, "reason": ""}  # 预检失败不阻断流程

    def identify_voiceprint(
        self, speaker_ids: List[str], audio_bytes: bytes
    ) -> Tuple[str, float]:
        """
        识别声纹

        Args:
            speaker_ids: 候选说话人ID列表
            audio_bytes: 音频字节数据

        Returns:
            Tuple[str, float]: (识别出的说话人ID, 相似度分数)
        """
        start_time = time.time()
        logger.info(f"开始声纹识别流程，候选说话人数量: {len(speaker_ids)}")

        audio_path = None
        try:
            # 音频大小验证
            if len(audio_bytes) < 1000:
                logger.warning("音频文件过小")
                return "", 0.0
            max_size = 5 * 1024 * 1024  # 5MB
            if len(audio_bytes) > max_size:
                logger.warning(f"音频文件过大: {len(audio_bytes)}字节，超过{max_size}字节限制")
                return "", 0.0

            # 音频质量预检
            quality = self._check_audio_quality(audio_bytes)
            if not quality["ok"]:
                logger.warning(f"音频质量预检不通过: {quality['reason']}，跳过识别")
                return "", 0.0
            if quality.get("snr_db", 60) < 10:
                logger.info(f"音频信噪比较低: {quality.get('snr_db', 0):.1f}dB，识别结果可能不可靠")

            # 处理音频文件（含降噪）
            audio_process_start = time.time()
            audio_path = audio_processor.ensure_16k_wav(audio_bytes)
            audio_process_time = time.time() - audio_process_start
            logger.debug(f"音频文件处理完成，耗时: {audio_process_time:.3f}秒")

            # 提取声纹特征
            extract_start = time.time()
            logger.debug("开始提取声纹特征...")
            test_emb = self.extract_voiceprint(audio_path)
            extract_time = time.time() - extract_start
            logger.debug(f"声纹特征提取完成，耗时: {extract_time:.3f}秒")

            # 获取候选声纹特征
            db_query_start = time.time()
            logger.debug("开始查询数据库获取候选声纹特征...")
            voiceprints = voiceprint_db.get_voiceprints(speaker_ids)
            db_query_time = time.time() - db_query_start
            logger.debug(
                f"数据库查询完成，获取到{len(voiceprints)}个声纹特征，耗时: {db_query_time:.3f}秒"
            )

            if not voiceprints:
                logger.info("未找到候选说话人声纹")
                return "", 0.0

            # 向量化计算相似度
            similarity_start = time.time()
            logger.debug("开始计算相似度...")
            candidate_names = list(voiceprints.keys())
            candidate_embs = np.stack(list(voiceprints.values()))
            test_emb_2d = test_emb.reshape(1, -1)
            sim_scores = self.calculate_similarity_matrix(test_emb_2d, candidate_embs)[0]  # [N]
            similarity_time = time.time() - similarity_start
            logger.debug(
                f"相似度计算完成，共计算{len(candidate_names)}个，耗时: {similarity_time:.3f}秒"
            )

            # 找到最佳匹配
            best_idx = int(np.argmax(sim_scores))
            match_name = candidate_names[best_idx]
            match_score = float(sim_scores[best_idx])

            # 检查是否超过阈值
            if match_score < self.similarity_threshold:
                logger.info(
                    f"未识别到说话人，最高分: {match_score:.4f}，阈值: {self.similarity_threshold}"
                )
                total_time = time.time() - start_time
                logger.info(f"声纹识别流程完成，总耗时: {total_time:.3f}秒")
                return "", match_score

            total_time = time.time() - start_time
            logger.info(
                f"识别到说话人: {match_name}, 分数: {match_score:.4f}, 总耗时: {total_time:.3f}秒"
            )
            return match_name, match_score

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"声纹识别异常，总耗时: {total_time:.3f}秒，错误: {e}")
            return "", 0.0
        finally:
            # 清理临时文件
            cleanup_start = time.time()
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)
            cleanup_time = time.time() - cleanup_start
            logger.debug(f"临时文件清理完成，耗时: {cleanup_time:.3f}秒")

    def identify_voiceprint_batch(
        self, speaker_ids: List[str], audio_bytes_list: List[bytes]
    ) -> List[dict]:
        """
        批量声纹识别：真正的批量推理 + 向量化相似度计算

        Args:
            speaker_ids: 候选说话人ID列表
            audio_bytes_list: 音频字节数据列表

        Returns:
            List[dict]: 每条音频的识别结果 [{index, speaker_id, score}]
        """
        start_time = time.time()
        logger.info(
            f"开始批量声纹识别，候选说话人: {len(speaker_ids)}，音频数量: {len(audio_bytes_list)}"
        )

        # 一次性加载所有候选声纹向量
        voiceprints = voiceprint_db.get_voiceprints(speaker_ids)
        if not voiceprints:
            logger.info("未找到候选说话人声纹")
            return [
                {"index": i, "speaker_id": "", "score": 0.0}
                for i in range(len(audio_bytes_list))
            ]

        # 构建候选矩阵 [N, D] 和名称映射
        candidate_names = list(voiceprints.keys())
        candidate_embs = np.stack(list(voiceprints.values()))

        # 阶段1：预处理所有音频，收集有效路径
        valid_indices = []  # 原始索引
        audio_paths = []
        results = [None] * len(audio_bytes_list)

        preprocess_start = time.time()
        for i, audio_bytes in enumerate(audio_bytes_list):
            try:
                if len(audio_bytes) < 1000:
                    logger.warning(f"批量识别[{i}]: 音频文件过小")
                    results[i] = {"index": i, "speaker_id": "", "score": 0.0}
                    continue
                max_size = 5 * 1024 * 1024
                if len(audio_bytes) > max_size:
                    logger.warning(f"批量识别[{i}]: 音频文件过大 {len(audio_bytes)}字节")
                    results[i] = {"index": i, "speaker_id": "", "score": 0.0}
                    continue

                audio_path = audio_processor.ensure_16k_wav(audio_bytes)
                audio_paths.append(audio_path)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"批量识别[{i}]音频预处理异常: {e}")
                results[i] = {"index": i, "speaker_id": "", "score": 0.0}

        preprocess_time = time.time() - preprocess_start
        logger.debug(f"音频预处理完成，有效{len(audio_paths)}条，耗时: {preprocess_time:.3f}秒")

        try:
            if audio_paths:
                # 阶段2：批量推理提取所有embedding
                extract_start = time.time()
                test_embs_list = self.extract_voiceprint_batch(audio_paths)
                extract_time = time.time() - extract_start
                logger.debug(f"批量特征提取完成，耗时: {extract_time:.3f}秒")

                # 阶段3：向量化相似度计算
                similarity_start = time.time()
                test_embs = np.stack(test_embs_list)  # [M, D]
                sim_matrix = self.calculate_similarity_matrix(test_embs, candidate_embs)  # [M, N]

                best_indices = np.argmax(sim_matrix, axis=1)  # [M]
                best_scores = sim_matrix[np.arange(len(best_indices)), best_indices]  # [M]
                similarity_time = time.time() - similarity_start
                logger.debug(f"向量化相似度计算完成，耗时: {similarity_time:.3f}秒")

                # 填充结果
                for j, orig_idx in enumerate(valid_indices):
                    best_name = candidate_names[best_indices[j]]
                    best_score = float(best_scores[j])
                    results[orig_idx] = {
                        "index": orig_idx,
                        "speaker_id": best_name,
                        "score": best_score,
                    }
                    logger.debug(
                        f"批量识别[{orig_idx}]: 最佳匹配={best_name}, 分数={best_score:.4f}"
                    )
        except Exception as e:
            logger.error(f"批量推理/相似度计算异常: {e}")
            for j, orig_idx in enumerate(valid_indices):
                if results[orig_idx] is None:
                    results[orig_idx] = {"index": orig_idx, "speaker_id": "", "score": 0.0}
        finally:
            # 清理所有临时文件
            for audio_path in audio_paths:
                audio_processor.cleanup_temp_file(audio_path)

        # 填充未处理的结果（防御性）
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {"index": i, "speaker_id": "", "score": 0.0}

        total_time = time.time() - start_time
        logger.info(
            f"批量声纹识别完成，处理{len(audio_bytes_list)}条(有效{len(audio_paths)}条)，总耗时: {total_time:.3f}秒"
        )
        return results

    def delete_voiceprint(self, speaker_id: str) -> bool:
        """
        删除声纹

        Args:
            speaker_id: 说话人ID

        Returns:
            bool: 删除是否成功
        """
        return voiceprint_db.delete_voiceprint(speaker_id)

    def get_voiceprint_count(self) -> int:
        """
        获取声纹总数

        Returns:
            int: 声纹总数
        """
        start_time = time.time()
        logger.info("开始获取声纹总数...")

        try:
            count = voiceprint_db.count_voiceprints()
            total_time = time.time() - start_time
            logger.info(f"声纹总数获取完成: {count}，耗时: {total_time:.3f}秒")
            return count
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"获取声纹总数失败，总耗时: {total_time:.3f}秒，错误: {e}")
            raise


# 全局声纹服务实例
voiceprint_service = VoiceprintService()
