import pymysql
from typing import Optional
from contextlib import contextmanager
from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """数据库连接管理类"""

    def __init__(self):
        self._connection: Optional[pymysql.Connection] = None
        self._connect()

    def _connect(self) -> None:
        """建立数据库连接"""
        try:
            mysql_config = settings.mysql
            password = (
                str(mysql_config["password"])
                if mysql_config["password"] is not None
                else ""
            )

            self._connection = pymysql.connect(
                host=mysql_config["host"],
                port=mysql_config["port"],
                user=mysql_config["user"],
                password=password,
                database=mysql_config["database"],
                charset="utf8mb4",
                autocommit=True,
                max_allowed_packet=16777216,  # 16MB
                connect_timeout=10,
                read_timeout=30,
                write_timeout=30,
            )
            logger.success("数据库连接成功")
        except Exception as e:
            logger.fail(f"数据库连接失败: {e}")
            raise

    def _ensure_connection(self) -> None:
        """确保数据库连接可用，断线自动重连"""
        try:
            if self._connection and self._connection.open:
                self._connection.ping(reconnect=True)
                return
        except Exception:
            logger.warning("数据库连接已断开，尝试重连...")
            self._connection = None

        self._connect()

    @contextmanager
    def get_cursor(self):
        """获取数据库游标的上下文管理器（带自动重连）"""
        self._ensure_connection()

        cursor = None
        try:
            cursor = self._connection.cursor()
            yield cursor
        except pymysql.OperationalError as e:
            # 连接级错误（如服务端断开），重连后重试一次
            error_code = e.args[0] if e.args else 0
            if error_code in (2006, 2013, 2014, 0):  # MySQL server has gone away / Lost connection
                logger.warning(f"数据库连接丢失(错误码{error_code})，重连重试...")
                if cursor:
                    cursor.close()
                    cursor = None
                self._connection = None
                self._connect()
                cursor = self._connection.cursor()
                # 注意：这里不能自动重试SQL，因为yield已经执行
                # 抛出异常让调用方重试
            if self._connection:
                self._connection.rollback()
            raise
        except Exception as e:
            logger.fail(f"数据库操作失败: {e}")
            if self._connection:
                self._connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def close(self) -> None:
        """关闭数据库连接"""
        if self._connection and self._connection.open:
            self._connection.close()
            logger.info("数据库连接已关闭")

    def __del__(self):
        """析构函数，确保连接被关闭"""
        try:
            self.close()
        except:
            pass


# 全局数据库连接实例
db_connection = DatabaseConnection()
