import os
import io
import uuid
import soundfile as sf
import librosa
import noisereduce as nr
import numpy as np
import time
from typing import Dict, Tuple
from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """音频处理工具类"""

    def __init__(self):
        self.target_sample_rate = settings.target_sample_rate
        self.tmp_dir = settings.tmp_dir
        # 确保临时目录存在
        os.makedirs(self.tmp_dir, exist_ok=True)
        # 启动时清理上次残留的临时文件（进程异常退出时未清理的）
        self._cleanup_stale_files()

    def ensure_16k_wav(self, audio_bytes: bytes, apply_denoise: bool = True) -> str:
        """
        将任意采样率的wav bytes转为16kHz wav临时文件。
        在内存中完成读取、重采样、截断、降噪，只写一次磁盘。

        Args:
            audio_bytes: 音频字节数据
            apply_denoise: 是否应用降噪处理（声纹识别建议开启，注册建议关闭）

        Returns:
            str: 临时文件路径
        """
        start_time = time.time()
        logger.debug(f"开始音频处理，输入大小: {len(audio_bytes)}字节")

        try:
            # 从内存读取音频（避免先写磁盘再读回）
            read_start = time.time()
            buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf)
            read_time = time.time() - read_start
            duration = len(data) / sr if sr > 0 else 0
            logger.debug(
                f"音频内存读取完成，采样率: {sr}Hz，时长: {duration:.2f}秒，耗时: {read_time:.3f}秒"
            )

            # 截断过长音频，防止大文件导致OOM
            max_duration = 30  # 秒
            if duration > max_duration:
                max_samples = int(max_duration * sr)
                if data.ndim == 1:
                    data = data[:max_samples]
                else:
                    data = data[:max_samples, :]
                logger.warning(f"音频时长 {duration:.1f}秒 超过限制，已截断至 {max_duration}秒")

            # 在内存中完成重采样
            if sr != self.target_sample_rate:
                resample_start = time.time()
                logger.debug(f"开始音频重采样: {sr}Hz -> {self.target_sample_rate}Hz")

                if data.ndim == 1:
                    data = librosa.resample(
                        data, orig_sr=sr, target_sr=self.target_sample_rate
                    )
                else:
                    data = np.vstack(
                        [
                            librosa.resample(
                                data[:, ch],
                                orig_sr=sr,
                                target_sr=self.target_sample_rate,
                            )
                            for ch in range(data.shape[1])
                        ]
                    ).T

                resample_time = time.time() - resample_start
                logger.debug(f"音频重采样完成，耗时: {resample_time:.3f}秒")
                sr = self.target_sample_rate

            # 降噪处理（重采样后、写磁盘前）
            if apply_denoise:
                data = self.denoise_audio(data, sr)

            # 只写一次磁盘（ModelScope pipeline 需要文件路径）
            write_start = time.time()
            tmp_path = os.path.join(self.tmp_dir, f"{uuid.uuid4().hex}.wav")
            sf.write(tmp_path, data, sr)
            write_time = time.time() - write_start
            logger.debug(f"音频写入完成，耗时: {write_time:.3f}秒")

            total_time = time.time() - start_time
            logger.debug(f"音频处理完成，总耗时: {total_time:.3f}秒")
            return tmp_path

        except Exception as e:
            # 清理临时文件
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            total_time = time.time() - start_time
            logger.error(f"音频处理失败，总耗时: {total_time:.3f}秒，错误: {e}")
            raise

    def validate_audio_file(self, audio_bytes: bytes) -> bool:
        """
        验证音频文件格式是否有效（简化版本，内存读取）

        Args:
            audio_bytes: 音频字节数据

        Returns:
            bool: 音频文件是否有效
        """
        start_time = time.time()
        logger.debug(f"开始音频文件验证，输入大小: {len(audio_bytes)}字节")

        try:
            # 从内存读取音频
            read_start = time.time()
            buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf)
            read_time = time.time() - read_start
            logger.debug(
                f"音频文件读取完成，采样率: {sr}Hz，数据长度: {len(data)}，耗时: {read_time:.3f}秒"
            )

            # 检查音频数据
            if len(data) == 0:
                logger.warning("音频文件为空")
                return False

            # 检查采样率
            if sr < 8000:  # 最低采样率要求
                logger.warning(f"采样率过低: {sr}Hz")
                return False

            # 检查音频时长（至少0.5秒，最多30秒）
            duration = len(data) / sr
            if duration < 0.5:
                logger.warning(f"音频时长过短: {duration:.2f}秒")
                return False
            elif duration > 30:
                logger.warning(f"音频时长过长: {duration:.2f}秒")
                return False

            total_time = time.time() - start_time
            logger.debug(
                f"音频验证通过: {duration:.2f}秒, {sr}Hz，总耗时: {total_time:.3f}秒"
            )
            return True

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"音频文件验证失败，总耗时: {total_time:.3f}秒，错误: {e}")
            return False

    def check_audio_quality(self, data: np.ndarray, sr: int) -> Dict:
        """
        检测音频质量指标，用于声纹识别前的质量预检

        Args:
            data: 音频数据（单声道 float）
            sr: 采样率

        Returns:
            dict: 质量指标，包含 ok, voice_duration, snr_db, clip_ratio, reason
        """
        if data.ndim > 1:
            data = data[:, 0]  # 取第一声道

        duration = len(data) / sr
        result = {
            "ok": True,
            "duration": round(duration, 2),
            "voice_duration": 0.0,
            "snr_db": 0.0,
            "clip_ratio": 0.0,
            "reason": "",
        }

        # 1. 整体 RMS 能量
        rms = np.sqrt(np.mean(data ** 2))
        if rms < 1e-5:
            result["ok"] = False
            result["reason"] = "音频几乎无声"
            return result

        # 2. 有效语音时长估算（基于短时能量）
        frame_len = int(0.025 * sr)  # 25ms 帧
        hop_len = int(0.01 * sr)     # 10ms 步长
        if len(data) > frame_len:
            frames = librosa.util.frame(data, frame_length=frame_len, hop_length=hop_len)
            frame_energy = np.sqrt(np.mean(frames ** 2, axis=0))
            energy_threshold = np.percentile(frame_energy, 30)  # 取30分位数作为噪声基线
            voice_frames = np.sum(frame_energy > energy_threshold * 2.0)
            result["voice_duration"] = round(voice_frames * hop_len / sr, 2)
        else:
            result["voice_duration"] = result["duration"]

        # 3. 信噪比估算 (SNR)
        if len(data) > frame_len:
            sorted_energy = np.sort(frame_energy)
            noise_est = np.mean(sorted_energy[:max(1, len(sorted_energy) // 5)])  # 最低20%作为噪声
            signal_est = np.mean(sorted_energy[-max(1, len(sorted_energy) // 5):])  # 最高20%作为信号
            if noise_est > 1e-8:
                result["snr_db"] = round(20 * np.log10(signal_est / noise_est), 1)
            else:
                result["snr_db"] = 60.0  # 噪声极低

        # 4. 削波检测
        result["clip_ratio"] = round(float(np.mean(np.abs(data) > 0.99)), 4)

        # 综合判断
        if result["voice_duration"] < 0.5:
            result["ok"] = False
            result["reason"] = f"有效语音时长过短: {result['voice_duration']:.2f}秒"
        elif result["snr_db"] < 3.0:
            result["ok"] = False
            result["reason"] = f"信噪比过低: {result['snr_db']:.1f}dB"
        elif result["clip_ratio"] > 0.1:
            result["ok"] = False
            result["reason"] = f"音频严重削波: {result['clip_ratio']:.1%}"

        return result

    def denoise_audio(self, data: np.ndarray, sr: int) -> np.ndarray:
        """
        频谱门控降噪，减少背景噪声对声纹提取的干扰

        Args:
            data: 音频数据（单声道 float）
            sr: 采样率

        Returns:
            np.ndarray: 降噪后的音频数据
        """
        start_time = time.time()
        try:
            if data.ndim > 1:
                data = data[:, 0]

            denoised = nr.reduce_noise(
                y=data,
                sr=sr,
                prop_decrease=0.6,       # 降噪强度 0-1，0.6 平衡降噪与语音保留
                n_fft=1024,
                stationary=False,        # 非平稳噪声模式，适应动态噪声
            )

            denoise_time = time.time() - start_time
            logger.debug(f"音频降噪完成，耗时: {denoise_time:.3f}秒")
            return denoised
        except Exception as e:
            logger.warning(f"音频降噪失败，使用原始音频: {e}")
            return data

    def _cleanup_stale_files(self) -> None:
        """启动时清理临时目录中的残留文件"""
        try:
            count = 0
            for f in os.listdir(self.tmp_dir):
                fp = os.path.join(self.tmp_dir, f)
                if os.path.isfile(fp) and f.endswith(".wav"):
                    os.remove(fp)
                    count += 1
            if count > 0:
                logger.info(f"已清理{count}个残留临时文件")
        except Exception as e:
            logger.warning(f"清理残留临时文件失败: {e}")

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        清理临时文件

        Args:
            file_path: 临时文件路径
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.debug(f"清理临时文件失败 {file_path}: {e}")


# 全局音频处理器实例
audio_processor = AudioProcessor()
